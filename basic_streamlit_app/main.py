import streamlit as st 
import pandas as pd #for data wrangling
import seaborn as sns  #for plotting
import matplotlib.pyplot as plt #for plotting

#Title
st.title("Exploring NBA Player Data Over Time")
st.write("Let's explore some NBA statistics and see how basketball has changed (and stayed the same) over the years.")
st.write("Note that this app only contains data from the 1996-97 season through the 2022-23 season.")

#Loading in data...
df = pd.read_csv("data/all_seasons.csv")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "NBA Demographic Trends", "Shooting Trends", "NBA Draft Pick Success", "Exploration"])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    # Title
    st.title("Exploring NBA Player Data Over Time")
    st.write("Let's explore some NBA statistics and see how basketball has changed (and stayed the same) over the years.")
    st.write("Note that this app only contains data from the 1996-97 season through the 2022-23 season.")

    st.write("Here is a table with all data. Feel free to explore!")
    st.dataframe(df)

    # Filtering the df by either team or season
    st.write("Filter to see your favorite team's stats for any time period!")
    teams = df["team_abbreviation"].unique()
    seasons = sorted(df["season"].unique())

    selected_team = st.selectbox("Select a team", ["All"] + list(teams), key="team_filter")

    # Season range slider
    season_range = st.select_slider(
        "Select season range",
        options=seasons,
        value=(seasons[0], seasons[-1])
    )

    # Filtering the data to include only dates in the slider
    filtered_df = df[
        (df["season"] >= season_range[0]) &
        (df["season"] <= season_range[1])
    ]

    # Filtering so that only a selected deam's dataframe is generated
    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["team_abbreviation"] == selected_team]

    st.dataframe(filtered_df)

# ---------------- TAB 2: TRENDS ----------------
with tab2:
    # 1. Country demographics over time

    #Calculating total number of players per season
    total_per_season = df.groupby("season").size()

    #Total American players per season
    usa_per_season = df[df["country"] == "USA"].groupby("season").size()

    #Computing percentage of American born players
    pct_usa = (usa_per_season / total_per_season * 100).reset_index(name="pct_usa")

    #Creating section titles
    st.subheader("How has NBA talent grown internationally?")
    st.write("Over the years, the NBA has made an effort to reach basketball talent outside the US. This has led to more modern NBA talent coming from elsewhere. Take a look at how this change has evolved over time:")
    st.subheader("% Of NBA Players Born in the USA Over Time")

    #Generating a percentage of American born players per year chart
    st.line_chart(pct_usa.set_index("season")["pct_usa"])

    # 2. Individual team demographics
    st.write("Let's explore how this trend has impacted individual teams:")

    #Generating a list of NBA teams for users to select
    teams = df["team_abbreviation"].unique()
    selected_team = st.selectbox("Select a team to see % of USA-born players", teams, key="pct_team")

    # Generating a button to reveal an insight about the Spurs' success with international players
    if st.button("Click me to reveal an interesting insight."):
        st.write("The 2012-13 and 2013-14 San Antonio Spurs (SAS) went to two straight NBA Finals, losing one in 7 games and winning the other. Select their team to see how they leveraged international talent during these years.")
    else:
        st.write(" ")

    #Filter df to only include selected team
    team_df = df[df["team_abbreviation"] == selected_team]

    #Filtering for USA born players on that team
    usa_team_df = team_df[team_df["country"] == "USA"]

    #Calculating percentage of USA born players on that selected team
    total_per_season = team_df.groupby("season").size()
    usa_per_season = usa_team_df.groupby("season").size()
    pct_usa_team = (usa_per_season / total_per_season * 100).reset_index(name="pct_usa")

    #bar chart of percentage of American players on that team
    st.subheader(f"Percentage of USA-born Players Over Time: {selected_team}")
    st.bar_chart(pct_usa_team.set_index("season")["pct_usa"])

# ---------------- TAB 3: EFFICIENCY ----------------
with tab3:
    #Calculating avg true shooting percentage for the whole league for each season
    ts_per_season = df.groupby("season")["ts_pct"].mean().reset_index()

    #Section Title and description
    st.subheader("NBA Player Shooting Efficiency Over Time")
    st.write("With rule changes, increased player talent, and evolving offensive strategies, the NBA has seemingly become more favorable to offense over time. Here, average true shooting percentage seems to have a slight upward trend.")
    st.subheader("Average True Shooting Percentage by Season")

    #Visualize league wide shooting efficiency over time
    st.line_chart(ts_per_season.set_index("season")["ts_pct"])

    #Team-specific TS%
    teams = sorted(df["team_abbreviation"].unique())

    #Seceltbox to explore team specific shooting trends
    selected_team_only = st.selectbox(
        "Select a team to view how its true shooting percentage has evolved over time",
        teams,
        key="ts_team_only"
    )

    #Filtering data for selected team
    team_only_df = df[df["team_abbreviation"] == selected_team_only]
    #calculating that team's true shooting percentage
    team_only_ts = team_only_df.groupby("season")["ts_pct"].mean().reset_index()

    #Visualizing ts% for that team
    st.subheader(f"True Shooting Percentage Over Time: {selected_team_only}")
    st.line_chart(team_only_ts.set_index("season")["ts_pct"])

    #Comparing that team to league average
    st.write("Let's compare how individual team averages compare to the league average over time")

    #Button to suggest that users explor Golden State Warriors shooting success
    if st.button("Click me for a suggestion on what to explore!"):
        st.write("See how the Golden State Warriors (GSW) compare to the rest of the league during their run to 5 straight NBA finals from 2014-15 to 2018-19.")
    else:
        st.write(" ")

    #Selecting a team to explore
    selected_team = st.selectbox("Select a team to compare TS%", teams, key="ts_team")

    #Filtering the data to only include selected team
    team_df = df[df["team_abbreviation"] == selected_team]
    #Calculating that team's ts%
    team_ts = team_df.groupby("season")["ts_pct"].mean().reset_index()

    #Plot comparison of selected team to league average
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(ts_per_season["season"], ts_per_season["ts_pct"], label="League Average")
    ax.plot(team_ts["season"], team_ts["ts_pct"], label=f"{selected_team}")
    ax.set_xlabel("Season")
    ax.set_ylabel("True Shooting %")
    ax.set_title("True Shooting Percentage Over Time")
    ax.set_xticklabels(ts_per_season["season"], rotation=90)
    ax.legend()

    st.pyplot(fig)

# ---------------- TAB 4: DRAFT ANALYSIS ----------------
with tab4:
    # Draft position vs success
    teams = sorted(df["team_abbreviation"].unique())

    #Removing undrafted players and converting draft positions to integers
    draft_df = df[df["draft_number"] != "Undrafted"].copy()
    draft_df["draft_number"] = draft_df["draft_number"].astype(int)

    #Section titles
    st.subheader("How Does Draft Position Affect NBA Success?")
    st.write("Unsuprisingly, there seems to be a clear correlation: the higher a player is drafted, the higher their NBA scoring average is.")
    st.subheader("Average Points by Draft Position")

    #Select range of players to analyze
    pick_range_pts = st.slider("Draft pick range (points per game chart)", 1, 60, (1, 60))

    #Generating an optional dropdown to explore draft postion for each team or for the whole league
    st.write("Click the dropdown below to explore average points by draft position for the entire league, or for an individual team.")

    teams_pts = sorted(df["team_abbreviation"].unique())
    selected_team_pts = st.selectbox(
        "Select team for Average Points by Draft Position",
        ["All"] + teams_pts,
        key="team_pts"
    )

    #Filter draft data for selected pick range
    filtered_df_pts = draft_df[
        draft_df["draft_number"].between(pick_range_pts[0], pick_range_pts[1])
    ]

    #Applying the team filter if user selects a specific team
    if selected_team_pts != "All":
        filtered_df_pts = filtered_df_pts[filtered_df_pts["team_abbreviation"] == selected_team_pts]

    #Computing average points for each draft postion
    avg_pts = (
        filtered_df_pts
        .groupby("draft_number")["pts"]
        .mean()
        .reset_index()
    )

    #Visualizing average points by draft position
    st.subheader("Average Points per Game by Draft Position")
    st.bar_chart(avg_pts.set_index("draft_number")["pts"])

    # Net rating analysis
    st.write("A player's plus minus measures how many points a player's team wins or loses by while they are on the floor. Since every player is weighted evenly, most net ratings fall below zero.")

    #Selecting a pick range to analyze
    pick_range_net = st.slider("Draft pick range (Net Rating Chart)", 1, 60, (1, 60))

    teams_net = sorted(df["team_abbreviation"].unique())
    selected_team_net = st.selectbox(
        "Select team for Average Net Rating by Draft Position",
        ["All"] + teams_net,
        key="team_net"
    )

    #Button to reveal insight about why the 57th pick has a positive net rating
    if st.button("Click me to reveal an interesting insight!"):
        st.write("You may notice that only three draft positions have a positive net rating. Unsuprisingly, two of these positions are the first and third overall picks. The other is the 57th overall pick. How? That can be explained by Manu Ginobili, the 57th overall pick in 1999 who played 16 seasons with the San Antonio Spurs, winning 4 championships.")
    else:
        st.write(" ")

    #Filtering for net rating analysis
    filtered_df_net = draft_df[
        draft_df["draft_number"].between(pick_range_net[0], pick_range_net[1])
    ]

    if selected_team_net != "All":
        filtered_df_net = filtered_df_net[filtered_df_net["team_abbreviation"] == selected_team_net]

    #Average net rating by draft positiion
    avg_net = (
        filtered_df_net
        .groupby("draft_number")["net_rating"]
        .mean()
    )

    #Visualizing avg net rating by draft position
    st.subheader("Average Net Rating by Draft Position")
    st.bar_chart(avg_net)

    # Undrafted players
    st.write("As the talent around the world has increased, more and more NBA talent has come from players that weren't even drafted!")

    #Make a df for just undrafted players
    undrafted_df = df[df["draft_number"] == "Undrafted"]
    seasons = sorted(undrafted_df["season"].unique())

    #Selecting season range slider
    season_range_undrafted = st.select_slider(
        "Select season range (Undrafted players)",
        options=seasons,
        value=(seasons[0], seasons[-1]),
        key="season_range_undrafted"
    )

    #Adding option to filter by team
    teams_undrafted = sorted(df["team_abbreviation"].unique())
    selected_team_undrafted = st.selectbox(
        "Select team for Undrafted Players Games Played",
        ["All"] + teams_undrafted,
        key="team_undrafted"
    )

    #Applying filters to generate df by season and team range
    filtered_undrafted = undrafted_df[
        (undrafted_df["season"] >= season_range_undrafted[0]) &
        (undrafted_df["season"] <= season_range_undrafted[1])
    ]

    if selected_team_undrafted != "All":
        filtered_undrafted = filtered_undrafted[filtered_undrafted["team_abbreviation"] == selected_team_undrafted]

    #Total games played by undrafted players per season
    gp_by_season = filtered_undrafted.groupby("season")["gp"].sum()

    #Visualizing games played by undrafted players
    st.subheader("Total Games Played by Undrafted Players Over Time")
    st.line_chart(gp_by_season)

# ---------------- TAB 5: PLAYER EXPLORER ----------------
with tab5:
    st.subheader("Finally, Explore How Your Favorite Player Has Performed Over Time")

    # Player selection
    players = sorted(df["player_name"].unique())
    selected_player = st.selectbox("Select a player", players, key="player_select")

    # Filter dataframe
    player_df = df[df["player_name"] == selected_player].sort_values("season")

    st.subheader(f"Stats Over Time: {selected_player}")

    # Points
    st.write("Points per game")
    st.line_chart(player_df.set_index("season")["pts"])

    # Rebounds
    st.write("Rebounds per game")
    st.line_chart(player_df.set_index("season")["reb"])

    # Assists
    st.write("Assists per game")
    st.line_chart(player_df.set_index("season")["ast"])

    # True Shooting %
    st.write("True Shooting Percentage")
    st.line_chart(player_df.set_index("season")["ts_pct"])





