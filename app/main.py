import pickle

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def preprocess_data():
    pass

def add_slide():
    st.sidebar.header("UCL Tournament Results")
    data = pd.read_csv("Data/ucl_stats.csv",encoding='latin1')

    all_items = data['team'].unique().tolist()

    # Text input as a search bar
    search_term = st.sidebar.text_input("Search for a Club:")

    # Filter the items based on the search
    filtered_items = [item for item in all_items if search_term.lower() in item.lower()] if search_term else all_items

    # Display filtered items in a selectbox
    selected_item = st.sidebar.selectbox("Select a Club:", filtered_items)

    input_dictionary = {}
    input_dictionary['team'] = selected_item

    # Show the selection
    st.sidebar.write("You selected:", selected_item)


    sliders_labels = [
        ("Matches Played", "match_played"),
        ("Matches Won", "wins"),
        ("Matches Drawn", "draws"),
        ("Matches Lost", "losts"),
        ("Goals Scored", "goals_scored"),
        ("Goals Conceded", "goals_conceded"),
        ("Goal Difference", "gd"),
        ("Group Points", "group_point")
    ]


    for label, col in sliders_labels:
        input_dictionary[col] =st.sidebar.slider(
            label=label,
            min_value=int(data[col].min()),
            max_value=int(data[col].max()),
            value=int(data[col].mean()),
            step=1
        )
    return input_dictionary



def get_radar_chart(input_dict):
    temp_dict = input_dict.copy()
    del temp_dict['team']
    data = pd.read_csv("Data/ucl_stats.csv",encoding='latin1')
    x = data.drop(['team','champions'],axis=1)
    scaled_dict = {}

    for key, value in temp_dict.items():
        min_val = x[key].min()
        max_val = x[key].max()
        scaled_dict[key] = (value - min_val) / (max_val - min_val)

    categories = ["Matches Played", "Matches Won","Matches Drawn","Matches Lost","Goals Scored","Goals Conceded","Goal Difference","Group Points"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            scaled_dict['match_played'],
            scaled_dict['wins'],
            scaled_dict['draws'],
            scaled_dict['losts'],
            scaled_dict['goals_scored'],
            scaled_dict['goals_conceded'],
            scaled_dict['gd'],
            scaled_dict['group_point'],
        ],
        theta=categories,
        fill='toself',
        name='Value'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False
    )
    return fig

def add_prediction(input_dictionary):
    model = pickle.load(open("models/model.pkl","rb"))
    transformer = pickle.load(open("models/tranformer.pkl","rb"))


    input_array = np.array(list(input_dictionary.values())).reshape(1, -1)



    input_array_trf = transformer.transform(pd.DataFrame(input_array,columns=['team', 'match_played', 'wins', 'draws', 'losts', 'goals_scored','goals_conceded', 'gd', 'group_point']))

    y_probs_all = model.predict_proba(input_array_trf)[:, 1]
    pred = (y_probs_all >= 0.09).astype(int)

    st.subheader("UCL score")
    if pred[0] == 0:
        st.error("Not Champion")
    else:
        st.success("Champion 🎉🎊🎉")


    st.write("Probability of being Champion:", round(model.predict_proba(input_array_trf)[0][1], 4))
    st.write("Probability of being Failure:", round(model.predict_proba(input_array_trf)[0][0], 4))


def main():
    st.set_page_config(
        page_title="UCL Champion Predictor",
        page_icon ="⚽",
        layout="wide",
        initial_sidebar_state='expanded'
    )


    input_dictionary = add_slide()

    with st.container():
        st.title("UCL Champion Predictor")
        st.write(
            "Please connect this app to predict this team will champion or not. This app uses a machine learning model to predict this. You can also adjust the measurements manually using the sliders in the sidebar.")

        col1, col2 = st.columns([3, 2])

        with col1:
            radar_chart = get_radar_chart(input_dictionary)
            st.plotly_chart(radar_chart)

        with col2:
            add_prediction(input_dictionary)


if __name__ == '__main__':
    main()