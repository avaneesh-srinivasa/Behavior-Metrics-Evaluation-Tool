import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Behavior Metrics Evaluation Tool", layout="wide")

st.title("Behavior Metrics Evaluation Tool")
st.write(
    "Evaluate ego vehicle behavior in a simple scenario replay (based on synthetic replay data) based on calculation and evaluation of behavior metrics."
)

st.markdown("""
### Scenario definition
**Scenario:** Ego vehicle follows a lead vehicle in the same lane on a straight road.  
The lead vehicle brakes suddenly. Ego vehicle reacts after a short delay and brakes.  

**Simplified assumptions:**  
- Straight single-lane road  
- One ego vehicle and one lead vehicle  
- Longitudinal motion only  
- Perfect state knowledge  
- Deterministic braking profiles  
""")

st.markdown("""
### High-level behavior evaluation steps
**Step 1:** Generate synthetic replay data based on user inputs  
**Step 2:** Compute behavior metrics such as gap, relative speed, and Time To Collision (TTC)  
**Step 3:** Evaluate the metrics over time against simple logic  
**Step 4:** Diagnose whether ego response was late, insufficient, overly aggressive, or acceptable  
""")

st.sidebar.header("Scenario Inputs")

ego_speed_kph = st.sidebar.slider("Ego vehicle speed (km/h)", 30, 100, 60)
lead_speed_kph = st.sidebar.slider("Lead vehicle speed (km/h)", 20, 100, 55)
initial_gap_m = st.sidebar.slider("Initial gap (m)", 10, 60, 25)

lead_brake_start_s = st.sidebar.slider("Lead vehicle brake start time (s)", 0.5, 5.0, 2.0, 0.1)
lead_decel = st.sidebar.slider("Lead vehicle deceleration (m/s²)", 1.0, 8.0, 4.0, 0.1)

ego_reaction_delay_s = st.sidebar.slider("Ego vehicle reaction delay (s)", 0.2, 2.0, 0.9, 0.1)
ego_decel = st.sidebar.slider("Ego vehicle deceleration (m/s²)", 1.0, 8.0, 4.0, 0.1)

dt = 0.2
sim_time_s = 10.0
vehicle_length_m = 4.5

ttc_caution = 3.0
ttc_unsafe = 1.5
late_response_limit = 0.8
late_ttc_limit = 2.5
aggressive_decel_limit = 5.0


def kph_to_mps(value):
    return value / 3.6


times = np.arange(0, sim_time_s + dt, dt)

ego_x = [0.0]
lead_x = [initial_gap_m + vehicle_length_m]

ego_v = [kph_to_mps(ego_speed_kph)]
lead_v = [kph_to_mps(lead_speed_kph)]

ego_a = [0.0]
lead_a = [0.0]

ego_brake_start_s = lead_brake_start_s + ego_reaction_delay_s

for i in range(1, len(times)):
    current_time = times[i]

    if current_time >= lead_brake_start_s:
        lead_acc = -lead_decel
    else:
        lead_acc = 0.0

    if current_time >= ego_brake_start_s:
        ego_acc = -ego_decel
    else:
        ego_acc = 0.0

    new_lead_v = lead_v[-1] + lead_acc * dt
    new_ego_v = ego_v[-1] + ego_acc * dt

    if new_lead_v < 0:
        new_lead_v = 0.0

    if new_ego_v < 0:
        new_ego_v = 0.0

    new_lead_x = lead_x[-1] + new_lead_v * dt
    new_ego_x = ego_x[-1] + new_ego_v * dt

    lead_a.append(lead_acc)
    ego_a.append(ego_acc)

    lead_v.append(new_lead_v)
    ego_v.append(new_ego_v)

    lead_x.append(new_lead_x)
    ego_x.append(new_ego_x)

df = pd.DataFrame(
    {
        "time_s": times,
        "ego_x_m": ego_x,
        "lead_x_m": lead_x,
        "ego_speed_mps": ego_v,
        "lead_speed_mps": lead_v,
        "ego_acc_mps2": ego_a,
        "lead_acc_mps2": lead_a,
    }
)

gap_list = []
relative_speed_list = []
ttc_list = []
ttc_band_list = []

for i in range(len(df)):
    gap = df.loc[i, "lead_x_m"] - df.loc[i, "ego_x_m"] - vehicle_length_m
    relative_speed = df.loc[i, "ego_speed_mps"] - df.loc[i, "lead_speed_mps"]

    if gap > 0 and relative_speed > 0:
        ttc = gap / relative_speed
    else:
        ttc = np.nan

    if gap <= 0:
        ttc_band = "Collision"
    elif pd.isna(ttc):
        ttc_band = "Not closing"
    elif ttc <= ttc_unsafe:
        ttc_band = "Unsafe"
    elif ttc <= ttc_caution:
        ttc_band = "Caution"
    else:
        ttc_band = "Comfortable"

    gap_list.append(gap)
    relative_speed_list.append(relative_speed)
    ttc_list.append(ttc)
    ttc_band_list.append(ttc_band)

df["gap_m"] = gap_list
df["relative_speed_mps"] = relative_speed_list
df["ttc_s"] = ttc_list
df["ttc_band"] = ttc_band_list

hazard_start_time = lead_brake_start_s

ego_response_time = np.nan
for i in range(len(df)):
    if df.loc[i, "ego_acc_mps2"] < -0.1:
        ego_response_time = df.loc[i, "time_s"]
        break

if pd.isna(ego_response_time):
    reaction_delay = np.nan
else:
    reaction_delay = ego_response_time - hazard_start_time

min_ttc = df["ttc_s"].min(skipna=True)
min_gap = df["gap_m"].min()
max_ego_decel = abs(df["ego_acc_mps2"].min())

collision = False
for value in df["gap_m"]:
    if value <= 0:
        collision = True
        break

unsafe_time_s = 0.0
for value in df["ttc_s"]:
    if not pd.isna(value) and value <= ttc_unsafe:
        unsafe_time_s = unsafe_time_s + dt

if collision:
    diagnosis = "Collision"
elif not pd.isna(min_ttc) and min_ttc <= ttc_unsafe:
    diagnosis = "Insufficient response"
elif not pd.isna(reaction_delay) and reaction_delay > late_response_limit and min_ttc < late_ttc_limit:
    diagnosis = "Late response"
elif max_ego_decel > aggressive_decel_limit and min_ttc > ttc_unsafe:
    diagnosis = "Overly aggressive response"
else:
    diagnosis = "Acceptable response"

st.subheader("Synthetic Replay Data")

show_df = df[
    [
        "time_s",
        "ego_x_m",
        "lead_x_m",
        "ego_speed_mps",
        "lead_speed_mps",
        "ego_acc_mps2",
        "lead_acc_mps2",
        "gap_m",
        "relative_speed_mps",
        "ttc_s",
        "ttc_band",
    ]
].round(3)


def highlight_metrics(row):
    styles = []
    for col in row.index:
        if col in ["gap_m", "relative_speed_mps", "ttc_s", "ttc_band"]:
            styles.append("background-color: #fff3cd")
        else:
            styles.append("")
    return styles


styled_table = show_df.style.apply(highlight_metrics, axis=1)
st.dataframe(styled_table, use_container_width=True, hide_index=True)
st.caption("Highlighted columns indicate computed behavior metrics.")

st.subheader("Behavior Diagnosis")

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"### {diagnosis}")
col2.metric("Minimum TTC", "N/A" if pd.isna(min_ttc) else f"{min_ttc:.2f} s")
col3.metric("Reaction Delay", "N/A" if pd.isna(reaction_delay) else f"{reaction_delay:.2f} s")
col4.metric("Max Ego Decel", f"{max_ego_decel:.2f} m/s²")

col5, col6 = st.columns(2)
col5.metric("Minimum Gap", f"{min_gap:.2f} m")
col6.metric("Time in Unsafe TTC Zone", f"{unsafe_time_s:.2f} s")

if diagnosis == "Collision":
    st.error("The replay results in a collision.")
elif diagnosis == "Insufficient response":
    st.warning("Ego response is not sufficient to maintain adequate TTC margin.")
elif diagnosis == "Late response":
    st.warning("Ego starts braking too late for the level of risk in the scenario.")
elif diagnosis == "Overly aggressive response":
    st.warning("Ego restores margin, but braking is very harsh.")
else:
    st.success("Ego response appears acceptable for this simplified behavior evaluation.")

st.markdown("**Logic used for behavior analysis**")

logic_df = pd.DataFrame(
    {
        "Priority": [1, 2, 3, 4, 5],
        "Behavior outcome": [
            "Collision",
            "Insufficient response",
            "Late response",
            "Overly aggressive response",
            "Acceptable response",
        ],
        "Logic used": [
            "Gap ≤ 0 at any timestep",
            f"Minimum TTC ≤ {ttc_unsafe:.1f} s",
            f"Reaction delay > {late_response_limit:.1f} s and minimum TTC < {late_ttc_limit:.1f} s",
            f"Max ego decel > {aggressive_decel_limit:.1f} m/s² and minimum TTC > {ttc_unsafe:.1f} s",
            "None of the above triggered",
        ],
    }
)

st.dataframe(logic_df, use_container_width=True, hide_index=True)

st.subheader("Behavior Metrics Over Time")

fig_gap = go.Figure()
fig_gap.add_trace(
    go.Scatter(
        x=df["time_s"],
        y=df["gap_m"],
        mode="lines+markers",
        name="Gap"
    )
)
fig_gap.add_vline(x=hazard_start_time, line_dash="dot", annotation_text="Lead brakes")
if not pd.isna(ego_response_time):
    fig_gap.add_vline(x=ego_response_time, line_dash="dot", annotation_text="Ego reacts")
fig_gap.update_layout(
    height=320,
    xaxis_title="Time (s)",
    yaxis_title="Gap (m)"
)
st.plotly_chart(fig_gap, use_container_width=True)

fig_rel = go.Figure()
fig_rel.add_trace(
    go.Scatter(
        x=df["time_s"],
        y=df["relative_speed_mps"],
        mode="lines+markers",
        name="Relative Speed"
    )
)
fig_rel.add_vline(x=hazard_start_time, line_dash="dot", annotation_text="Lead brakes")
if not pd.isna(ego_response_time):
    fig_rel.add_vline(x=ego_response_time, line_dash="dot", annotation_text="Ego reacts")
fig_rel.update_layout(
    height=320,
    xaxis_title="Time (s)",
    yaxis_title="Relative Speed (m/s)"
)
st.plotly_chart(fig_rel, use_container_width=True)

fig_ttc = go.Figure()
fig_ttc.add_trace(
    go.Scatter(
        x=df["time_s"],
        y=df["ttc_s"],
        mode="lines+markers",
        name="TTC"
    )
)
fig_ttc.add_hline(y=ttc_caution, line_dash="dash", annotation_text="Caution")
fig_ttc.add_hline(y=ttc_unsafe, line_dash="dash", annotation_text="Unsafe")
fig_ttc.add_vline(x=hazard_start_time, line_dash="dot", annotation_text="Lead brakes")
if not pd.isna(ego_response_time):
    fig_ttc.add_vline(x=ego_response_time, line_dash="dot", annotation_text="Ego reacts")
fig_ttc.update_layout(
    height=320,
    xaxis_title="Time (s)",
    yaxis_title="TTC (s)"
)
st.plotly_chart(fig_ttc, use_container_width=True)
