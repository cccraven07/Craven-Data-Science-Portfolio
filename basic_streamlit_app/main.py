import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Title
st.title("Exploring NBA Player Data Over Time")
st.write("Let's explore some NBA statistics and see how basketball has changed (and stayed the same) over the years.")
st.write("Note that this app only contains data from the 1996-97 season through the 2022-23 season.")

#Loading in data...
df = pd.read_csv("data/all_seasons.csv")

st.write("Here is a table with all data. Feel free to explore!")
st.dataframe(df)

#Filtering the df by either team or season: 
st.write("Filter to see your favorite team's stats for any time period!")
teams = df["team_abbreviation"].unique()
seasons = sorted(df["season"].unique())

selected_team = st.selectbox("Select a team", ["All"] + list(teams), key="team_filter")

#Season range slider
season_range = st.select_slider(
    "Select season range",
    options=seasons,
    value=(seasons[0], seasons[-1])
)

filtered_df = df[
    (df["season"] >= season_range[0]) &
    (df["season"] <= season_range[1])
]

#Team filter
if selected_team != "All":
    filtered_df = filtered_df[filtered_df["team_abbreviation"] == selected_team]

st.dataframe(filtered_df)


#1. First, Let's Explore how country demographics of the NBA have changed over time
total_per_season = df.groupby("season").size()
usa_per_season = df[df["country"] == "USA"].groupby("season").size()
pct_usa = (usa_per_season / total_per_season * 100).reset_index(name="pct_usa")


st.subheader("1. How has NBA talent grown internationally?")
st.write("Over the years, the NBA has made an effort to reach basketball talent outside the US. This has led to more modern NBA talent coming from elsewhere. Take a look at how this change has evolved over time:")
st.subheader("% Of NBA Players Born in the USA Over Time")

st.line_chart(pct_usa.set_index("season")["pct_usa"])


#2. Individual team demographics
teams = df["team_abbreviation"].unique()
st.write("Let's explore how this trend has impacted individual teams:")

selected_team = st.selectbox("Select a team to see % of USA-born players", teams, key="pct_team")

#Highlighting the San Antonio Spurs international success
if st.button("Click me to a reveal an interesting insight."):
    st.write("The 2012-13 and 2013-14 San Antonio Spurs (SAS) went to two straight NBA Finals, losing one in 7 games and winning the other. Select their team to see how leveraged international talent during these years:")
else:
    st.write(" ")

#Filtering data frames for USA born players on each team
team_df = df[df["team_abbreviation"] == selected_team]
usa_team_df = team_df[team_df["country"] == "USA"]

#Filtering so that it can be written as a percentage
total_per_season = team_df.groupby("season").size()
usa_per_season = usa_team_df.groupby("season").size()
pct_usa_team = (usa_per_season / total_per_season * 100).reset_index(name="pct_usa")

st.subheader(f"Percentage of USA-born Players Over Time: {selected_team}")

#Chart of USA born players for each team
st.bar_chart(pct_usa_team.set_index("season")["pct_usa"])


#4. Let's see how true shooting percentage has changed over time
ts_per_season = df.groupby("season")["ts_pct"].mean().reset_index()

st.subheader("2. NBA Player Shooting Efficiency Over Time")
st.write("With rule changes, increased player talent, and evolving offensive strategies, the NBA has seemingly become more favorable to offense over time. Here, average true shooting percentage seems to have a slight upward trend.")
st.subheader("Average True Shooting Percentage by Season")

#Line chart of total true shooting percentage
st.line_chart(ts_per_season.set_index("season")["ts_pct"])

#Average true shooting percentage across all seasons
ts_per_season = df.groupby("season")["ts_pct"].mean().reset_index()

#Teams dropdown
teams = sorted(df["team_abbreviation"].unique())

selected_team_only = st.selectbox("Select a team to view how its true shooting percentage has evolved over time", teams, key="ts_team_only")

#Filtering true shooting percentage by team
team_only_df = df[df["team_abbreviation"] == selected_team_only]
team_only_ts = team_only_df.groupby("season")["ts_pct"].mean().reset_index()

#Graph of true shooting percentage for individual teams
st.subheader(f"True Shooting Percentage Over Time: {selected_team_only}")
st.line_chart(team_only_ts.set_index("season")["ts_pct"])

#Comparing teams with the entire league on the same graph
st.write("Let's compare how individual team averages compare to the league average over time")
if st.button("Click me for a suggestion on what to explore!"):
    st.write("See how the Golden State Warriors (GSW) compare to the rest of the league during their run to 5 straight NBA finals from 2014-15 to 2018-19.")
else:
    st.write(" ")

selected_team = st.selectbox("Select a team to compare TS%", teams, key="ts_team")

# Team true shooting percentage per season
team_df = df[df["team_abbreviation"] == selected_team]
team_ts = team_df.groupby("season")["ts_pct"].mean().reset_index()

# Plotting...
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(ts_per_season["season"], ts_per_season["ts_pct"], color="blue", label="League Average")
ax.plot(team_ts["season"], team_ts["ts_pct"], color="red", label=f"{selected_team}")
ax.set_xlabel("Season")
ax.set_ylabel("True Shooting %")
ax.set_title("True Shooting Percentage Over Time")
ax.set_xticklabels(ts_per_season["season"], rotation=90)
ax.legend()
st.pyplot(fig)

#6. Let's see how draft position correlates with success
teams = sorted(df["team_abbreviation"].unique())

# Draft pick data frames. Getting rid of undrafted values
draft_df = df[df["draft_number"] != "Undrafted"].copy()
draft_df["draft_number"] = draft_df["draft_number"].astype(int)

st.subheader("3. How Does Draft Position Affect NBA Success?")
st.write("Unsuprisingly, there seems to be a clear correlation: the higher a player is drafted, the higher their NBA scoring average is.")
st.subheader("Average Points by Draft Postion")

#To select draft pick range
pick_range_pts = st.slider("Draft pick range (points per game chart)", 1, 60, (1, 60))

st.write("Click the dropbox below to explore average points by draft postion for the entire league, or for an individual team.")

# Dropdown for points graph
teams_pts = sorted(df["team_abbreviation"].unique())
selected_team_pts = st.selectbox(
    "Select team for Average Points by Draft Position",
    ["All"] + teams_pts,
    key="team_pts"
)

#To filter on draft pick range
filtered_df_pts = draft_df[
    draft_df["draft_number"].between(pick_range_pts[0], pick_range_pts[1])
]

if selected_team_pts != "All":
    filtered_df_pts = filtered_df_pts[filtered_df_pts["team_abbreviation"] == selected_team_pts]

avg_pts = (
    filtered_df_pts
    .groupby("draft_number")["pts"]
    .mean()
    .reset_index()
)

#Bar chart of average points per game by draft postition
st.subheader("Average Points per Game by Draft Position")
st.bar_chart(avg_pts.set_index("draft_number")["pts"])

#7. Draft pick vs Net Rating
st.write("A player's plus minus measures how much a points a player's team wins or loses by while they are on the floor. Since every player is weighted evenly (i.e., players who play very few minutes count as much as players who play a lot) the majority of net ratings are below zero.)")

#Slider for draft pick range
pick_range_net = st.slider("Draft pick range (Net Rating Chart)", 1, 60, (1, 60))

#Dropdown for net rating graph
teams_net = sorted(df["team_abbreviation"].unique())
selected_team_net = st.selectbox(
    "Select team for Average Net Rating by Draft Position",
    ["All"] + teams_net,
    key="team_net"
)

#Explaining the outlier at 57... Manu Ginobili was insane!
if st.button("Click me to reveal an interesting insight!"):
    st.write("You may notice that only three draft positions have a positive net rating. Unsuprisingly, two of these positions are the first and third overall picks. The other is the 57th overall pick. How? That can be explained by Manu Ginobili, the 57th overall pick in 1999 who played 16 seasons with the San Antonio Spurs, winning 4 championships.")
else:
    st.write(" ")


filtered_df_net = draft_df[
    draft_df["draft_number"].between(pick_range_net[0], pick_range_net[1])
]

if selected_team_net != "All":
    filtered_df_net = filtered_df_net[filtered_df_net["team_abbreviation"] == selected_team_net]

avg_net = (
    filtered_df_net
    .groupby("draft_number")["net_rating"]
    .mean()
)

st.subheader("Average Net Rating by Draft Position")
st.bar_chart(avg_net)

#8. Undrafted players over time
st.write("As the talent around the world has increased, more and more NBA talent has come from players that weren't even drafted!")

#filtering to include just undrafted players
undrafted_df = df[df["draft_number"] == "Undrafted"]
seasons = sorted(undrafted_df["season"].unique())

season_range_undrafted = st.select_slider(
    "Select season range (Undrafted players)",
    options=seasons,
    value=(seasons[0], seasons[-1]),
    key="season_range_undrafted"
)

#Dropdown for Undrafted graph
teams_undrafted = sorted(df["team_abbreviation"].unique())
selected_team_undrafted = st.selectbox(
    "Select team for Undrafted Players Games Played",
    ["All"] + teams_undrafted,
    key="team_undrafted"
)


filtered_undrafted = undrafted_df[
    (undrafted_df["season"] >= season_range_undrafted[0]) &
    (undrafted_df["season"] <= season_range_undrafted[1])
]

if selected_team_undrafted != "All":
    filtered_undrafted = filtered_undrafted[filtered_undrafted["team_abbreviation"] == selected_team_undrafted]

gp_by_season = filtered_undrafted.groupby("season")["gp"].sum()

st.subheader("Total Games Played by Undrafted Players Over Time")
st.line_chart(gp_by_season)

#Finally, See how your favorite player has perfomred over time
st.subheader("Finally, feel free to explore how your favorite player's stats have evolved over time!")

#To get individual players...
players = sorted(df["player_name"].unique())
selected_player = st.selectbox("Select a player", players, key="player_select")

# Filter dataframe for selected player
player_df = df[df["player_name"] == selected_player].sort_values("season")

st.subheader(f"Stats Over Time: {selected_player}")

#Points
st.write("Points per game")
st.line_chart(player_df.set_index("season")["pts"])

#Rebounds
st.write("Rebounds per game")
st.line_chart(player_df.set_index("season")["reb"])

#Assists
st.write("Assists per game")
st.line_chart(player_df.set_index("season")["ast"])

#True Shooting Percentage
st.write("True Shooting Percentage per game")
st.line_chart(player_df.set_index("season")["ts_pct"])





