ROOT_URL = "https://data.wa.aemo.com.au/public/market-data/wemde/"

# Both caseInputData and every dispatchSolution report-type subfolder share
# this current/previous shape, so download.py just walks these two per dataset.
DATASET_FOLDERS = ("current", "previous")

# Convenience names for download.py / the parsers. dispatchSolution has several
# report-type subfolders (dispatchData, preDispatchData, weekAheadDispatchData,
# and an "-AvailableCapacity" variant of each) - dispatchData is the one
# actually being parsed for now.
CASE_INPUT_DATASET = "caseInputData"
DISPATCH_SOLUTION_DATASET = "dispatchSolution/dispatchData"


esr_codes = sorted(
    [
        "ALINTA_WGP_ESR1",
        "COLLIE_BESS2",
        "COLLIE_ESR1",
        "COLLIE_ESR4",
        "COLLIE_ESR5",
        "KWINANA_ESR1",
        "KWINANA_ESR2",
        "SBSOLAR1_CUNDERDIN_PV1",
        "TESLA_PICTON_G1",
        "PRDSO_WALPOLE_HG1",
    ]
)

battery_codes = [
    "COLLIE_BESS2",
    "COLLIE_ESR1",
    "COLLIE_ESR4",
    "COLLIE_ESR5",
    "KWINANA_ESR1",
    "KWINANA_ESR2",
]


# this information is obtained from
# https://explore.openelectricity.org.au/facilities/wem/?selected=KWINANA_ESR&tech=battery_discharging&status=operating
battery_capacity_MWh = {
    "ALINTA_WGP_ESR1": 200,
    "COLLIE_BESS2": 1300,
    "COLLIE_ESR1": 800,
    "COLLIE_ESR4": 1200,
    "COLLIE_ESR5": 1200,
    "KWINANA_ESR1": 200,
    "KWINANA_ESR2": 900,
}

battery_capacity_MW = {
    "ALINTA_WGP_ESR1": 100,  # 2hr battery
    "COLLIE_BESS2": 300,  # ~4hr battery
    "COLLIE_ESR1": 200,  # ~4hr battery
    "COLLIE_ESR4": 250,  # ~4.5hr battery
    "COLLIE_ESR5": 250,  # ~4.5hr battery
    "KWINANA_ESR1": 100,  # 2hr battery
    "KWINANA_ESR2": 225,  # ~4hr battery
}


# fixed daily shading window used on every event-day plot ("Peak ESROI"),
# replacing the old per-event start/end times below
PEAK_ESROI_START = "17:30"
PEAK_ESROI_END = "21:30"

system_stress_events = [
    {"date": "2024-12-10", "event_start_time": "15:00", "event_end_time": "20:30", "highest_stress_time": "18:35"},
    {"date": "2024-12-11", "event_start_time": "15:00", "event_end_time": "20:30", "highest_stress_time": "18:35"},
    {"date": "2025-01-20", "event_start_time": "15:00", "event_end_time": "20:30", "highest_stress_time": "18:35"},
    {"date": "2025-01-21", "event_start_time": "11:25", "event_end_time": "20:30", "highest_stress_time": "17:50"},
    {"date": "2025-01-23", "event_start_time": "13:00", "event_end_time": "20:30", "highest_stress_time": "18:50"},
    {"date": "2025-03-06", "event_start_time": "15:00", "event_end_time": "22:00", "highest_stress_time": "18:20"},
    {"date": "2024-01-13", "event_start_time": "16:00", "event_end_time": "20:00", "highest_stress_time": None},
    {"date": "2025-08-25", "event_start_time": "17:45", "event_end_time": "23:40", "highest_stress_time": None},
    {"date": "2026-05-30", "event_start_time": None, "event_end_time": None, "highest_stress_time": None},
]
