PLOT_PARAMETERS = {
    "delta TM": {
        "x_col": "Sum_Single_Mutant_delta_TM",
        "y_col": "delta TM",
        "title": r"$\Delta T_{m, \text{mutant}}$ vs $\sum\Delta T_{m, \text{sm}}$",
        "xlabel": r"Single Mutants: $\sum\Delta T_m$",
        "ylabel": r"Mutant: $\Delta T_m$",
        "ref_lines": (0,),
        "use_log": False,
    },
    "Binding Multiplicative": {
        "x_col": "KD_FC_mutant",
        "y_col": "KD_FC_sm_prod",
        "title": r"Binding FC: $\frac{KD_\text{wt}}{KD_\text{mut}}$ vs $\prod \frac{\text{wt}}{KD_\text{sm}}$",
        "xlabel": r"Single Mutants: $\left(\prod \frac{\text{wt}}{KD_\text{sm}}\right)$",
        "ylabel": r"Mutant: $\left(\frac{KD_\text{wt}}{KD_\text{mut}}\right)$",
        "ref_lines": (1,),  # Automatically converts to log10(1) = 0
        "use_log": False,
    },
    "EC50 Multiplicative": {
        "x_col": "EC50(prod(wt/sm))",
        "y_col": "EC50(wt/mut)",
        "title": r"EC50 FC: $\frac{EC50_{wt}}{EC50_{mut}}$ vs $\prod\frac{EC50_{wt}}{EC50_{sm}}$",
        "xlabel": r"Single Mutants: $\left(\prod \frac{EC50_{wt}}{EC50_{sm}}\right)$",
        "ylabel": r"Mutant: $\left(\frac{EC50_{wt}}{EC50_{mut}}\right)$",
        "ref_lines": (1,),  # Automatically converts to log10(1) = 0
        "use_log": False,
    },
}