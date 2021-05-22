import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import gzip


st.set_page_config(page_title="New COVID-19 cases predictions", page_icon="😷", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
    with open(modelfile, 'rb') as f:
	    loaded_model = pickle.load(f)
	    return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:DARKSALMON;text-align:left;"> New COVID-19 cases predictions  😷 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    with st.beta_expander(" ℹ️ Information", expanded=True):
            st.write("""
                🦠 Last year, the world had been hit by a pandemic, that nobody was prepared for. The various measures taken by the differents governements have been ingenious, urgent and specific to each country, even if all was decided around a common base. One year after, it's time to check what was the best measures taken, and what we should do in the future, thanks to the data we got. In this case, we didn't considerate the vaccination, because it's obviously the best way to contain the pandemic, however, a vaccin is not always discover, so we need to know what is the best combinaison of measures in case of a possible new pandemic like the COVID-19.
                """)
            '''
            ## How does it work ❓ 
            🔮 Complete all the parameters and the machine learning model will predict the number of new case 7 days after
            '''

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
            '''
            ## DETAILS 
            🏫 SCHOOL -> School Closure policy:

            0. No measures
            0. Recommend closing or all schools
            0. Require closing (only some levels or categories)
            0. Require closing all levels
            
            🏗 WORK -> Work Policy:
            
            0. No measures
            0. Recommend closing (or recommend work from home)
            0. Require closing/work from home for some sectors or categories of workers
            0. Require closing/or work from home for all-but-essential workplaces (eg doctors)
            
            🏟 EVENTS -> Event Policy:
            
            0. No measures
            0. Recommend cancelling
            0. Require cancelling
            
            🍻 GATHERINGS -> Private & Public Gatherings Policy:
            
            0. No restrictions
            0. Restrictions on very large gatherings (the limit is above 1000 people)
            0. Restrictions on gatherings between 101-1000 people
            0. Restrictions on gatherings between 11-100 people
            0. Restrictions on gatherings of 10 people or less
            
            🚎 TRANSPORTATION -> Public Transportation Policy:
            
            0. No measures
            0. Recommend closing (or significantly reduce volume/route/means of transport available)
            0. Require closing (or prohibit most citizens from using it)
            
            🏡 AT HOME -> Stay at home policy:
            
            0. No measures
            0. Recommend not leaving house
            0. Require not leaving house with exceptions for daily exercise, grocery shopping, and 'essential' trips
            0. Require not leaving house with minimal exceptions (eg allowed to leave once a week, or only one person can leave at a time, etc)
            
            🇫🇷 NATIONAL -> Restriction on internal movement :
            
            0. No measures
            0. Recommend not to travel between regions/cities
            0. Internal movement restrictions in place
            
            🌏 INTERNATIONAL -> International travel controls:
            
            0. No restrictions
            0. Screening arrivals
            0. Quarantine arrivals from some or all regions
            0. Ban arrivals from some regions
            0. Ban on all regions or total border closure
            
            🎙 INFORMATION -> Public Information campaigns:
            
            0. No Covid-19 public information campaign
            0. Public officials urging caution about Covid-19
            0. Coordinated public information campaign (eg across traditional and social media)
            
            💉 TESTING -> Testing Policy:
            
            0. No testing policy
            0. Only those who both (a) have symptoms AND (b) meet specific criteria (eg key workers, admitted to hospital, came into contact with a known case, returned from overseas)
            0. Testing of anyone showing Covid-19 symptoms
            0. Open public testing (eg "drive through" testing available to asymptomatic people)
            
            🔬 TRACING ->Contact tracing policy:
            
            0. No contact tracing
            0. Limited contact tracing; not done for all cases
            0. Comprehensive contact tracing; done for all identified cases
            
            🦠 NEW CASES -> The number of new cases:
            
            The number of new infected of the COVID-19 this day
            '''



    with col2:
        st.subheader("Find out how many new COVID-19 cases will happend in 7 days")
        School = st.number_input("School", 0,3)
        Work = st.number_input("work", 0, 3)
        Events = st.number_input("Events", 0, 2)
        Gatherings = st.number_input("Gatherings", 0, 4)
        Transportation = st.number_input("Transportation", 0, 2)
        AtHome = st.number_input("At Home", 0, 3)
        National = st.number_input("National", 0, 2)
        International = st.number_input("International", 0, 4)
        Information= st.number_input("Information", 0, 2)
        Testing = st.number_input("Testing", 0, 3)
        Tracing = st.number_input("Tracing", 0, 3)
        NewCases = st.number_input("New Cases", 1, 500000)


        feature_list = [School,
        Work,
        Events,
        Gatherings,
        Transportation,
        AtHome,
        National,
        International,
        Information,
        Testing,
        Tracing, 
        NewCases,]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('lr_full_model.pkl')
            prediction = loaded_model.predict(single_pred)
            col2.write('''
		    ## Results 🔍 
		    ''')
            col2.success(f"{prediction.item()} new cases are predicted in 7 days by our A.I.")

      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    st.warning("Note: This A.I application is for educational/demo purposes only and cannot be relied upon.")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()