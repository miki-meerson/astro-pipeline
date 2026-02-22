import os
import numpy as np
from plotly.subplots import make_subplots
import json
from pathlib import Path
import datetime


def mc_shifts_fig(mc_shifts):
    fig = make_subplots \
            (
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0,
            row_titles=["horizontal <br> pixels shifts", "vertical <br> pixels shifts"],
            y_title='pixels', x_title="Frames [#]"
        )

    fig.update_layout(height=400, width=1000)
    fig.add_scatter(name='x-shifts', x=np.arange(len(mc_shifts[:, 1])), y=mc_shifts[:, 1],
                    line=dict(width=0.4, color='blue'), showlegend=True, row=1, col=1)
    fig.add_scatter(name='y-shifts', x=np.arange(len(mc_shifts[:, 0])), y=mc_shifts[:, 0],
                    line=dict(width=0.4, color='blue'), showlegend=True, row=2, col=1)
    return fig


def serialize_value(v):
    # primitives
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v

    # numpy
    if isinstance(v, (np.integer, np.floating)):
        return v.item()

    if isinstance(v, np.ndarray):
        return v.tolist()

    # pathlib
    if isinstance(v, Path):
        return str(v)

    # datetime
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()

    # fallback
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)   # last resort