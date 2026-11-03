-- WEMDE database: the whole structure. Running this file wipes and recreates every table.
-- Change the structure here, commit it to git, then rebuild.
--
-- Naming:   issued_at = primaryDispatchInterval (when the run was made for)
--           valid_at  = dispatchInterval        (the interval a row is about)
-- Units:    MW unless the column name says otherwise. Signs as AEMO gives them (withdrawal < 0).
-- Not stored: constraints (Case Input and Dispatch Solution), dfcmData, <FAC>.pdiSentOut,
--             rocofControlRequirements, contingencySolution, trapeziumAdjustments,
--             essPreProcessingConditionFlag.
--             The raw files are kept, so these can be added later.

DROP VIEW  IF EXISTS dispatch_actual, battery_soc_live, battery_projection, price_setters, runs_preferred;
DROP TABLE IF EXISTS
    solution_contingency_facilities, solution_contingencies, solution_price_setting,
    solution_facilities, solution_intervals,
    case_rcoq, case_violation_penalties, case_parameters,
    case_offer_tranches, case_offers, case_unconstrained_forecasts, case_scada, case_forecasts, cases,
    runs, facility_scada, loss_factors, facility_standing, scada_tags, facilities, loaded_files
CASCADE;


-- =====================================================================
-- 1. Lookup tables (small, change rarely)
-- =====================================================================

-- One row = one facility. Filled from any source the first time a code is seen.
-- facility_type is your judgment, loaded from facility_types.csv (not in AEMO's JSON).
CREATE TABLE facilities (
    facility_code    text PRIMARY KEY,
    facility_type    text NOT NULL DEFAULT 'unknown',  -- battery, wind, solar, gas, coal, biogas, waste, hydro, hybrid, load, unknown
    type_confidence  text,                             -- code (from the name), guess, checked
    type_note        text,
    is_battery       boolean GENERATED ALWAYS AS (facility_type = 'battery') STORED
);

-- One row = one version of a facility's standing data (Case Input registrationData).
-- The same version repeats in every case, so it is stored once.
CREATE TABLE facility_standing (
    facility_code        text NOT NULL REFERENCES facilities,
    effective_from       timestamptz NOT NULL,     -- effectiveDispatchIntervalFrom
    registration_status  text,
    facility_class       text,
    storage_constraints  boolean,
    nol                  boolean,
    inertia              double precision,
    tau_factor           double precision,
    unconstrained_forecast_source text,
    PRIMARY KEY (facility_code, effective_from)
);

-- One row = one facility's loss factors for one financial year (Case Input facilityLossFactors).
CREATE TABLE loss_factors (
    facility_code   text NOT NULL REFERENCES facilities,
    financial_year  smallint NOT NULL,             -- 2027 = 1 Jul 2026 to 30 Jun 2027
    dlf             double precision,
    tlf             double precision,
    PRIMARY KEY (facility_code, financial_year)
);

-- One row = one SCADA tag, split into its parts.
-- 'COLLIE_ESR1.energy.sentOut' -> kind facility, facility_code COLLIE_ESR1, measure energy.sentOut
-- 'dispatchCondition.demand'   -> kind system,   facility_code NULL,        measure dispatchCondition.demand
-- 'ALB.LINE.MBR_ALB_81.MW'     -> kind network,  facility_code NULL,        measure ALB.LINE.MBR_ALB_81.MW, element_type LINE
CREATE TABLE scada_tags (
    tag_id           smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tag              text NOT NULL UNIQUE,
    kind             text NOT NULL CHECK (kind IN ('facility', 'system', 'network')),
    facility_code    text REFERENCES facilities,
    substation       text,                         -- network tags: first part, e.g. ALB
    element_type     text,                         -- network tags: second part: LINE, TRANS, GEN, CB, SPS, ISOL, SVC, FDR, ...
    measure          text NOT NULL,
    value_data_type  text                          -- valueDataType: fixed per tag, so stored here
);


-- =====================================================================
-- 2. Runs: one row = one engine run (one schedule, one scenario, one issue time)
--    Case Input and Dispatch Solution rows both point here.
-- =====================================================================

-- AEMO sometimes republishes a run as a correction file with a suffix after the timestamp,
-- e.g. ReferenceDispatchSolution_202401011200_AffectedDispatchInterval.json (~0.6% of intervals).
-- Both are kept; the runs_preferred view picks the correction.
CREATE TABLE runs (
    run_id        integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- short stand-in for the columns below
    schedule      text NOT NULL,           -- 'Dispatch', 'Pre-Dispatch', 'Week-Ahead'
    scenario      text NOT NULL,           -- 'Reference', ...
    issued_at     timestamptz NOT NULL,    -- primaryDispatchInterval
    file_variant  text NOT NULL DEFAULT 'regular',  -- 'regular', or the filename suffix of a correction
    UNIQUE (schedule, scenario, issued_at, file_variant)
);


-- =====================================================================
-- 3. Case Input (the engine's inputs)
-- =====================================================================

-- One row = one case: one run, one interval.
CREATE TABLE cases (
    run_id     integer NOT NULL REFERENCES runs ON DELETE CASCADE,
    valid_at   timestamptz NOT NULL,
    status     text,
    minutes_since_primary  smallint,     -- constraintLookUp.solver.minutesSincePrimary
    interval_length_min    smallint,     -- constraintLookUp.solver.intervalLength
    -- markets.<service>.baseForecastRequirement / overrideForecastRequirement
    req_energy                    double precision,
    req_energy_override           double precision,
    req_regulation_raise          double precision,
    req_regulation_raise_override double precision,
    req_regulation_lower          double precision,
    req_regulation_lower_override double precision,
    req_contingency_raise         double precision,
    req_contingency_raise_override double precision,
    req_contingency_lower         double precision,
    req_contingency_lower_override double precision,
    req_rocof                     double precision,
    req_rocof_override            double precision,
    -- marketServicesPriceFlag.<service>PriceAtCap
    energy_price_at_cap             boolean,
    regulation_raise_price_at_cap   boolean,
    regulation_lower_price_at_cap   boolean,
    contingency_raise_price_at_cap  boolean,
    contingency_lower_price_at_cap  boolean,
    rocof_price_at_cap              boolean,
    PRIMARY KEY (run_id, valid_at)
);

-- Long format: one row = one forecast value from constraintLookUp.
-- name keeps AEMO's key without the 'forecast.' prefix, e.g. 'energy-forecastOperationalDemand-expected'.
CREATE TABLE case_forecasts (
    run_id    integer NOT NULL,
    valid_at  timestamptz NOT NULL,
    name      text NOT NULL,
    value     double precision,
    PRIMARY KEY (run_id, valid_at, name),
    FOREIGN KEY (run_id, valid_at) REFERENCES cases ON DELETE CASCADE
);

-- Long format: one row = one SCADA reading in one case. Every tag except <FAC>.pdiSentOut.
CREATE TABLE case_scada (
    run_id        integer NOT NULL,
    valid_at      timestamptz NOT NULL,
    tag_id        smallint NOT NULL REFERENCES scada_tags,
    value         double precision,           -- scada[].value, when it is a number
    value_text    text,                       -- scada[].value, when it isn't
    quality_flag  text,
    data_source   text,
    as_at         timestamptz,                -- asAtTimeStamp
    PRIMARY KEY (run_id, valid_at, tag_id),
    FOREIGN KEY (run_id, valid_at) REFERENCES cases ON DELETE CASCADE
);

-- One row = one facility's unconstrained forecast in one case (unconstrainedForecast).
CREATE TABLE case_unconstrained_forecasts (
    run_id         integer NOT NULL,
    valid_at       timestamptz NOT NULL,
    facility_code  text NOT NULL REFERENCES facilities,
    injection_mw   double precision,
    withdrawal_mw  double precision,
    PRIMARY KEY (run_id, valid_at, facility_code),
    FOREIGN KEY (run_id, valid_at) REFERENCES cases ON DELETE CASCADE
);

-- One row = one facility's offer for one service in one case (markets.<service>.facilities).
-- Energy offers fill the capacity, ramp and fsip columns; FCESS offers fill the enablement columns.
CREATE TABLE case_offers (
    run_id         integer NOT NULL,
    valid_at       timestamptz NOT NULL,
    service        text NOT NULL,            -- energy, regulationRaise, ..., rocof
    facility_code  text NOT NULL REFERENCES facilities,
    submission_id    text,
    submission_code  text,
    -- energy
    max_injection_mw     double precision,
    max_withdrawal_mw    double precision,
    unconstrained_injection_mw   double precision,
    unconstrained_withdrawal_mw  double precision,
    inflexible           boolean,
    max_up_ramp          double precision,
    max_down_ramp        double precision,
    up_ramp_used         double precision,
    down_ramp_used       double precision,
    fsip_t1              double precision,   -- fsip: fast start inflexibility profile
    fsip_t2              double precision,
    fsip_t3              double precision,
    fsip_t4              double precision,
    fsip_minimum_load    double precision,
    -- FCESS: offered and used trapezium
    maximum_capacity     double precision,
    enablement_min       double precision,
    enablement_min_used  double precision,
    low_breakpoint       double precision,
    low_breakpoint_used  double precision,
    high_breakpoint      double precision,
    high_breakpoint_used double precision,
    enablement_max       double precision,
    enablement_max_used  double precision,
    PRIMARY KEY (run_id, valid_at, service, facility_code),
    FOREIGN KEY (run_id, valid_at) REFERENCES cases ON DELETE CASCADE
);

-- One row = one price-quantity step of one offer.
CREATE TABLE case_offer_tranches (
    run_id         integer NOT NULL,
    valid_at       timestamptz NOT NULL,
    service        text NOT NULL,
    facility_code  text NOT NULL,
    tranche        smallint NOT NULL,
    quantity_mw      double precision,
    submitted_price  double precision,
    lfa_price        double precision,
    capacity_type    text,            -- IN-SERVICE / AVAILABLE ...
    fuel_type        text,
    notice_time      integer,
    PRIMARY KEY (run_id, valid_at, service, facility_code, tranche),
    FOREIGN KEY (run_id, valid_at, service, facility_code) REFERENCES case_offers ON DELETE CASCADE
);

-- Long format: one row = one global setting in one case (parameters[]).
CREATE TABLE case_parameters (
    run_id    integer NOT NULL,
    valid_at  timestamptz NOT NULL,
    name      text NOT NULL,
    value     double precision,
    PRIMARY KEY (run_id, valid_at, name),
    FOREIGN KEY (run_id, valid_at) REFERENCES cases ON DELETE CASCADE
);

-- One row = one constraint violation penalty multiplier in one case (constraintViolationParameters[]).
CREATE TABLE case_violation_penalties (
    run_id    integer NOT NULL,
    valid_at  timestamptz NOT NULL,
    variable  text NOT NULL,
    price     double precision,      -- multiplier x the energy price ceiling
    PRIMARY KEY (run_id, valid_at, variable),
    FOREIGN KEY (run_id, valid_at) REFERENCES cases ON DELETE CASCADE
);

-- One row = one facility's Reserve Capacity Obligation Quantity in one case (rcmData[]).
CREATE TABLE case_rcoq (
    run_id         integer NOT NULL,
    valid_at       timestamptz NOT NULL,
    facility_code  text NOT NULL REFERENCES facilities,
    rcoq           double precision,
    PRIMARY KEY (run_id, valid_at, facility_code),
    FOREIGN KEY (run_id, valid_at) REFERENCES cases ON DELETE CASCADE
);


-- =====================================================================
-- 4. Dispatch Solution (the engine's outputs)
-- =====================================================================

-- Wide: one row = one interval of one run. System-level market numbers.
CREATE TABLE solution_intervals (
    run_id     integer NOT NULL REFERENCES runs ON DELETE CASCADE,
    valid_at   timestamptz NOT NULL,
    run_uuid         uuid,              -- runId
    dispatch_run_id  uuid,              -- dispatchRunId
    status           text,
    solve_status     text,
    objective        double precision,
    fast_start       boolean,
    intervention     boolean,
    ocd              boolean,           -- over-constrained dispatch run
    solve_start      timestamptz,
    solve_finish     timestamptz,
    -- prices ($/MWh for energy, $/MW for FCESS)
    price_energy               double precision,
    price_regulation_raise     double precision,
    price_regulation_lower     double precision,
    price_contingency_raise    double precision,
    price_contingency_lower    double precision,
    price_rocof                double precision,
    -- priceSetting[].isMarketServiceCapped
    energy_capped              boolean,
    regulation_raise_capped    boolean,
    regulation_lower_capped    boolean,
    contingency_raise_capped   boolean,
    contingency_lower_capped   boolean,
    rocof_capped               boolean,
    -- fcessMarketClearingPriceCeilings[]
    ceiling_regulation_raise   double precision,
    ceiling_regulation_lower   double precision,
    ceiling_contingency_raise  double precision,
    ceiling_contingency_lower  double precision,
    ceiling_rocof              double precision,
    -- dispatchTotal
    total_injection            double precision,
    total_withdrawal           double precision,
    total_regulation_raise     double precision,
    total_regulation_lower     double precision,
    total_contingency_raise    double precision,
    total_contingency_lower    double precision,
    total_rocof                double precision,
    -- marketServiceRequirements
    req_energy                 double precision,
    req_regulation_raise       double precision,
    req_regulation_lower       double precision,
    req_contingency_raise      double precision,
    req_contingency_lower      double precision,
    req_rocof                  double precision,
    -- inServiceQuantities
    in_service_injection          double precision,
    in_service_withdrawal         double precision,
    in_service_regulation_raise   double precision,
    in_service_regulation_lower   double precision,
    in_service_contingency_raise  double precision,
    in_service_contingency_lower  double precision,
    in_service_rocof              double precision,
    -- availableQuantities
    available_injection           double precision,
    available_withdrawal          double precision,
    available_regulation_raise    double precision,
    available_regulation_lower    double precision,
    available_contingency_raise   double precision,
    available_contingency_lower   double precision,
    available_rocof               double precision,
    -- marketShortfalls
    deficit_energy             double precision,
    deficit_regulation_raise   double precision,
    deficit_regulation_lower   double precision,
    deficit_contingency_raise  double precision,
    deficit_contingency_lower  double precision,
    deficit_rocof              double precision,
    PRIMARY KEY (run_id, valid_at)
);

-- Wide: one row = one facility in one interval of one run.
-- schedule[].facilitySchedule (targets, pivoted to one column per service)
-- + facilityScheduleDetails + dispatchCaps.
-- Loaded: every facility at the primary interval, batteries for the whole horizon (rule in extract.py).
CREATE TABLE solution_facilities (
    run_id         integer NOT NULL,
    valid_at       timestamptz NOT NULL,
    facility_code  text NOT NULL REFERENCES facilities,
    -- targets (MW)
    energy                  double precision,
    regulation_raise        double precision,
    regulation_lower        double precision,
    contingency_raise       double precision,
    contingency_lower       double precision,
    rocof                   double precision,
    dispatch_cap            double precision,
    -- details
    initial_mw               double precision,   -- = Case Input energy.sentOut at the primary interval (battery power without Case Input)
    projected_charge_mwh     double precision,   -- projectedChargeLevel (<= 0), end of interval
    projected_discharge_mwh  double precision,   -- projectedDischargeLevel (>= 0), end of interval
    contingency              double precision,
    facility_risk            double precision,
    facility_lower_risk      double precision,
    load_contingency         double precision,
    nol_demand               double precision,
    congestion_rental        double precision,
    est_fcess_uplift         double precision,
    -- performance factors
    contingency_raise_pf     double precision,
    contingency_lower_pf     double precision,
    regulation_raise_pf      double precision,
    regulation_lower_pf      double precision,
    rocof_pf                 double precision,
    -- fast start
    fast_start_flag               boolean,
    fast_start_initial_mode       smallint,
    fast_start_initial_mode_time  integer,
    fast_start_target_mode        smallint,
    fast_start_target_mode_time   integer,
    PRIMARY KEY (run_id, valid_at, facility_code),
    FOREIGN KEY (run_id, valid_at) REFERENCES solution_intervals ON DELETE CASCADE
);

-- One row = one contribution to one service's price (priceSetting[].contributions[]).
-- price = sum(by_how_much x coefficient) over the rows for that service.
CREATE TABLE solution_price_setting (
    run_id         integer NOT NULL,
    valid_at       timestamptz NOT NULL,
    service        text NOT NULL,             -- the price being explained
    variable       text NOT NULL,             -- contributingVariable, e.g. KWINANA_ESR2_energy_Tranche_02
    by_how_much    double precision,          -- MW change for +1 MW of requirement
    coefficient    double precision,          -- objectiveCoefficient ($/MW)
    facility_code  text,                      -- split from variable; NULL for slack variables
    variable_service text,                    -- split from variable, e.g. energy, contingencyRaise
    tranche        smallint,                  -- split from variable
    PRIMARY KEY (run_id, valid_at, service, variable),
    FOREIGN KEY (run_id, valid_at) REFERENCES solution_intervals ON DELETE CASCADE
);

-- One row = one defined contingency (definedContingency[] and definedLoadContingency[]).
-- Loaded for the primary interval only (the lists are large and repeat across the horizon).
CREATE TABLE solution_contingencies (
    run_id          integer NOT NULL,
    valid_at        timestamptz NOT NULL,
    kind            text NOT NULL CHECK (kind IN ('supply', 'load')),
    contingency_id  text NOT NULL,
    value           double precision,         -- valueOfContingency
    load_contribution double precision,       -- load contingencies only
    PRIMARY KEY (run_id, valid_at, kind, contingency_id),
    FOREIGN KEY (run_id, valid_at) REFERENCES solution_intervals ON DELETE CASCADE
);

-- One row = one facility's contribution to one load contingency (definedLoadContingency[].associatedFacilities[]).
CREATE TABLE solution_contingency_facilities (
    run_id          integer NOT NULL,
    valid_at        timestamptz NOT NULL,
    kind            text NOT NULL,
    contingency_id  text NOT NULL,
    facility_code   text NOT NULL,
    service         text NOT NULL,
    contribution    double precision,
    PRIMARY KEY (run_id, valid_at, kind, contingency_id, facility_code, service),
    FOREIGN KEY (run_id, valid_at, kind, contingency_id) REFERENCES solution_contingencies ON DELETE CASCADE
);


-- =====================================================================
-- 5. Measured data (CSV folders)
-- =====================================================================

-- One row = one facility's metered energy in one 5-min Dispatch Interval.
-- Check the column names against the CSV header before loading.
CREATE TABLE facility_scada (
    valid_at       timestamptz NOT NULL,     -- start of the Dispatch Interval
    facility_code  text NOT NULL REFERENCES facilities,
    mwh            double precision,          -- x 12 = average MW
    PRIMARY KEY (valid_at, facility_code)
);


-- =====================================================================
-- 6. Bookkeeping
-- =====================================================================

-- One row = one raw file that has been loaded.
CREATE TABLE loaded_files (
    file_name       text PRIMARY KEY,
    issue_id        text,                     -- dispatchDataIssueID
    transaction_id  text,                     -- transactionId
    n_rows          integer NOT NULL,
    loaded_at       timestamptz NOT NULL DEFAULT now()
);


-- =====================================================================
-- 7. Indexes for common questions
-- =====================================================================

CREATE INDEX ON runs (issued_at);
CREATE INDEX ON solution_facilities (facility_code, valid_at);
CREATE INDEX ON solution_price_setting (facility_code);
CREATE INDEX ON case_scada (tag_id, valid_at);


-- =====================================================================
-- 8. Views (saved queries)
-- =====================================================================

-- One run per (schedule, scenario, issued_at): the correction file if there is one, else the regular file.
-- The other views build on this, so analysis never sees both versions of a run.
CREATE VIEW runs_preferred AS
SELECT DISTINCT ON (schedule, scenario, issued_at) *
FROM runs
ORDER BY schedule, scenario, issued_at, (file_variant = 'regular'), run_id DESC;

-- What actually got dispatched: the first interval of each Dispatch run, no look-ahead.
CREATE VIEW dispatch_actual AS
SELECT r.scenario, s.*
FROM solution_intervals s
JOIN runs_preferred r USING (run_id)
WHERE r.schedule = 'Dispatch'
  AND s.valid_at = r.issued_at;

-- Who set each price in each actual dispatch interval.
CREATE VIEW price_setters AS
SELECT r.scenario, r.issued_at, p.service, p.variable, p.facility_code, p.variable_service, p.tranche,
       p.by_how_much, p.coefficient, p.by_how_much * p.coefficient AS contribution
FROM solution_price_setting p
JOIN runs_preferred r USING (run_id)
WHERE r.schedule = 'Dispatch'
  AND p.valid_at = r.issued_at;

-- Live battery state of charge from the Case Input, one row per battery per case.
CREATE VIEW battery_soc_live AS
SELECT
    r.issued_at,
    s.valid_at,
    t.facility_code,
    max(s.value) FILTER (WHERE t.measure = 'dischargeLevelV2') AS discharge_mwh,
    max(s.value) FILTER (WHERE t.measure = 'chargeLevelV2')    AS charge_mwh,
    max(s.value) FILTER (WHERE t.measure = 'chargeLevel')      AS charge_level_v1,
    max(s.value) FILTER (WHERE t.measure = 'energy.sentOut')   AS sent_out_mw
FROM case_scada s
JOIN scada_tags t USING (tag_id)
JOIN runs_preferred r USING (run_id)
JOIN facilities f USING (facility_code)
WHERE f.is_battery
  AND r.schedule = 'Dispatch'
GROUP BY r.issued_at, s.valid_at, t.facility_code;

-- Battery targets and projected charge across each run's horizon.
CREATE VIEW battery_projection AS
SELECT
    r.schedule,
    r.scenario,
    r.issued_at,
    sf.valid_at,
    EXTRACT(EPOCH FROM sf.valid_at - r.issued_at) / 60 AS lead_min,
    sf.facility_code,
    sf.energy,
    sf.regulation_raise,
    sf.regulation_lower,
    sf.projected_discharge_mwh,
    sf.projected_charge_mwh,
    sf.projected_discharge_mwh - sf.projected_charge_mwh AS usable_mwh
FROM solution_facilities sf
JOIN runs_preferred r USING (run_id)
JOIN facilities f USING (facility_code)
WHERE f.is_battery;
