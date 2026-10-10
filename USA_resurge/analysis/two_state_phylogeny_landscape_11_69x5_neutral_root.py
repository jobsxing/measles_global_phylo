from __future__ import annotations

import core_dta_visualization as core
from two_state_phylogeny_clockwise90_width4_a4height import draw_clockwise_rotated_tree


def main() -> None:
    prepared = core.prepare_tree(
        core.GLOBAL["hipstr"],
        core.GLOBAL["trees"],
        core.GLOBAL["traits"],
        "state2",
        "globalUSA_HIPSTR_posterior_state_recoding.csv",
    )
    traits = core.load_traits(core.GLOBAL["traits"])
    draw_clockwise_rotated_tree(
        prepared,
        traits.date.min(),
        traits.date.max(),
        figure_size=(14.69, 5.0),
        stem="2state_phylogeny_full_timescale_landscape_11_69x5_neutral_root",
        title="Two-state DTA phylogeny: USA vs global non-USA",
        title_x=.055,
        title_y=.975,
        title_fontsize=17,
        view_label="Vertical root-to-tip (RTT) view",
        subtitle_x=.055,
        subtitle1_y=.875,
        subtitle2_y=.815,
        subtitle_fontsize=10.7,
        legend_anchor=(.050, .750),
        legend_columns=4,
        legend_fontsize=8.5,
        axes_left=.115,
        axes_right=.975,
        axes_top=.625,
        axes_bottom=.055,
        date_label_x=.025,
        date_label_y=.34,
        date_label_fontsize=13,
    )


if __name__ == "__main__":
    main()
