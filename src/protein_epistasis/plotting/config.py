PLOT_PARAMETERS = {
    "delta TM": {
        "x_col": "Sum_Single_Mutant_delta_TM",
        "y_col": "delta TM",
        "title": r"$\Delta T_{\text{m}, \text{mutant}}(^\circ\text{C}) \text{ vs } \sum\Delta T_{\text{m}, \text{sm}} (^\circ\text{C})$",
        "xlabel": r"$\text{Single Mutants: } \sum\Delta T_{\text{m}}(^\circ\text{C})$",
        "ylabel": r"$\text{Mutant: } \Delta T_{\text{m}}(^\circ\text{C})$",
        "ref_lines": (0,),
        "use_log": False,
    },
    "Binding Multiplicative": {
        "x_col": "KD_FC_mutant",
        "y_col": "KD_FC_sm_prod",
        "title": r"$\text{Binding FC: } \frac{\text{KD}_{\text{wt}}}{\text{KD}_{\text{mut}}} \text{ vs } \prod \frac{\text{KD}_\text{wt}}{\text{KD}_{\text{sm}}}$",
        "xlabel": r"$\text{Single Mutants: } \left(\prod \frac{\text{KD}_\text{wt}}{\text{KD}_{\text{sm}}}\right)$",
        "ylabel": r"$\text{Mutant: } \left(\frac{\text{KD}_{\text{wt}}}{\text{KD}_{\text{mut}}}\right)$",
        "ref_lines": (1,),  # Automatically converts to log10(1) = 0
        "use_log": False,
    },
    "EC50 Multiplicative": {
        "x_col": "EC50(prod(wt/sm))",
        "y_col": "EC50(wt/mut)",
        "title": r"$\text{EC50 FC: } \frac{\text{EC50}_{\text{wt}}}{\text{EC50}_{\text{mut}}} \text{ vs } \prod\frac{\text{EC50}_{\text{wt}}}{\text{EC50}_{\text{sm}}}$",
        "xlabel": r"$\text{Single Mutants: } \left(\prod \frac{\text{EC50}_{\text{wt}}}{\text{EC50}_{\text{sm}}}\right)$",
        "ylabel": r"$\text{Mutant: } \left(\frac{\text{EC50}_{\text{wt}}}{\text{EC50}_{\text{mut}}}\right)$",
        "ref_lines": (1,),  # Automatically converts to log10(1) = 0
        "use_log": False,
    },
}

