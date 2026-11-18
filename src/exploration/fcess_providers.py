"""Who provides FCESS? Each battery's share of the MW enabled in each FCESS service, by month.

    python -m exploration.fcess_providers
"""

from analysis.fcess import monthly_enabled_shares
from analysis.price_setting import OTHER_FACILITIES
from tools.constants import battery_codes
from tools.plot_style import UNIT_COLORS
from visualisation.plot_fcess import plot_monthly_fcess_shares

# Jul 2024: the first month any battery was enabled for FCESS (KWINANA_ESR1)
START = "2024-07-01"
# Nov 2026 holds only the first day and a bit of data, so stop before it
END = "2026-11-01"


def main():
    shares = monthly_enabled_shares(battery_codes)
    # tz-aware index vs date string: compared in its own tz
    shares = {service: s.loc[(s.index >= START) & (s.index < END)] for service, s in shares.items()}

    plot_monthly_fcess_shares(
        shares,
        colors={**UNIT_COLORS, OTHER_FACILITIES: "0.75"},
        title="Who provides FCESS: share of MW enabled in each service, by month",
        filename="fcess/fcess_share_by_battery_monthly.png",
    )
    for service, s in shares.items():
        print(f"\n{service}\n{s.assign(batteries=s[battery_codes].sum(axis=1)).round(1).to_string()}")


if __name__ == "__main__":
    main()
