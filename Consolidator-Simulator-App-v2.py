from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st
import pandas as pd
import numpy as np
import googlemaps
import plotly.graph_objects as go
import ast
from streamlit_folium import st_folium
import folium
from PIL import Image

def str_to_tuple(coord_str):
    return ast.literal_eval(coord_str)

def plot_interactive_map(df):
    # Initial setup
    center_coordinates = [54.5260, 15.2551]  # Centered on Europe
    zoom_level = 4

    # Create the map object only once
    if "map" not in st.session_state:
        st.session_state.map = folium.Map(location=center_coordinates, zoom_start=zoom_level)
        st.session_state.clicked_location = None  # Initialize clicked location state

        # Add markers for Load and Delivery points from the DataFrame
        for i, row in df.iterrows():
            load_lat, load_lon = str_to_tuple(row['Load_Coord'])
            del_lat, del_lon = str_to_tuple(row['Del_Coord'])

            # Add Load marker
            folium.Marker(
                location=[load_lat, load_lon],
                popup=f"Load Location {i + 1}: {row['Load_Coord']}",
                icon=folium.Icon(color="green")
            ).add_to(st.session_state.map)

            # Add Delivery marker
            folium.Marker(
                location=[del_lat, del_lon],
                popup=f"Delivery Location {i + 1}: {row['Del_Coord']}",
                icon=folium.Icon(color="red")
            ).add_to(st.session_state.map)

    # Render the existing map and capture clicked locations
    map_data = st_folium(st.session_state.map, width=700, height=500)

    # Debugging: Print map data
    # st.write("Map Data:", map_data)  # Use st.write instead of print for Streamlit

    # Handle marker for the last clicked location
    if map_data and 'last_clicked' in map_data:
        last_clicked = map_data['last_clicked']

        # Check if the last_clicked contains lat and lng
        if last_clicked and 'lat' in last_clicked and 'lng' in last_clicked:
            lat = last_clicked['lat']
            lon = last_clicked['lng']

            # Update clicked location in session state
            st.session_state.clicked_location = (lat, lon)

            # Clear previous markers from the map (except Load and Delivery markers)
            st.session_state.map = folium.Map(location=center_coordinates, zoom_start=zoom_level,key='map')  # Reinitialize the map

            # Re-add Load and Delivery markers
            for i, row in df.iterrows():
                load_lat, load_lon = str_to_tuple(row['Load_Coord'])
                del_lat, del_lon = str_to_tuple(row['Del_Coord'])

                # Add Load marker
                folium.Marker(
                    location=[load_lat, load_lon],
                    popup=f"Load Location {i + 1}: {row['Load_Coord']}",
                    icon=folium.Icon(color="green")
                ).add_to(st.session_state.map)

                # Add Delivery marker
                folium.Marker(
                    location=[del_lat, del_lon],
                    popup=f"Delivery Location {i + 1}: {row['Del_Coord']}",
                    icon=folium.Icon(color="red")
                ).add_to(st.session_state.map)

            # Add the clicked location marker to the existing map
            if st.session_state.clicked_location:
                lat, lon = st.session_state.clicked_location
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Selected Location: ({lat}, {lon})",
                    icon=folium.Icon(color="blue")
                ).add_to(st.session_state.map)

    # Finally, render the updated map with all markers (Load, Delivery, and clicked location)
    st_folium(st.session_state.map, width=0.1, height=0.1,key='map')
    if st.session_state.clicked_location==None:
        return 0.0,0.0
    return st.session_state.clicked_location


def plot_delivery_map(df):
    fig = go.Figure()

    # Function to convert string coordinates to tuples
    def str_to_tuple(coord_str):
        return ast.literal_eval(coord_str)
    
    # Initialize min/max values for latitudes and longitudes
    min_lat, max_lat = float('inf'), float('-inf')
    min_lon, max_lon = float('inf'), float('-inf')

    # Plotting Load and Delivery locations
    for i, row in df.iterrows():
        load_lat, load_lon = str_to_tuple(row['Load_Coord'])
        del_lat, del_lon = str_to_tuple(row['Del_Coord'])
        
        # Update min/max latitudes and longitudes for bounding box
        min_lat = min(min_lat, load_lat, del_lat)
        max_lat = max(max_lat, load_lat, del_lat)
        min_lon = min(min_lon, load_lon, del_lon)
        max_lon = max(max_lon, load_lon, del_lon)

        # Add lines connecting Load to Delivery
        fig.add_trace(go.Scattermapbox(
            lon=[load_lon, del_lon], lat=[load_lat, del_lat],
            mode='lines', line=dict(width=2, color='blue'),
            hoverinfo='none'
        ))
        
        # Add Load Point
        fig.add_trace(go.Scattermapbox(
            lon=[load_lon], lat=[load_lat],
            mode='markers',
            marker=go.scattermapbox.Marker(size=10, color='green'),
            text=f"Load: {row['Load_full_add']}",
            hoverinfo='text'
        ))

        # Add Delivery Point
        fig.add_trace(go.Scattermapbox(
            lon=[del_lon], lat=[del_lat],
            mode='markers',
            marker=go.scattermapbox.Marker(size=10, color='red'),
            text=f"Delivery: {row['Delivery_full_add']}",
            hoverinfo='text'
        ))

    # Calculate the center of the map based on the bounding box
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Adjust zoom level to fit all points in the bounding box
    zoom_level = 5  # Start with a reasonable zoom level
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    if lat_range > 0 or lon_range > 0:
        zoom_level = min(8 - lat_range * 10, 8 - lon_range * 10)

    # Update map layout for aesthetics and fitting
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=go.layout.mapbox.Center(lat=center_lat, lon=center_lon),
            zoom=3
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig

def plot_result(inbound_cost, stor_handling, outbound_cost, existing_cost):
    # Calculate the total new cost
    new_cost_total = inbound_cost + stor_handling + outbound_cost

    # Create the figure
    fig = go.Figure()

    # Add a bar for Existing Cost
    fig.add_trace(go.Bar(
        x=['Existing Cost'],  # Single category for Existing Cost
        y=[existing_cost],  # Existing cost value
        name='Existing Cost',
        marker_color='#4B8BBE',  # Blue for existing cost
        text=[existing_cost],  # Add existing cost label
        textposition='inside',  # Show text inside the bar
    ))

    # Add a stacked bar for Inbound Cost
    fig.add_trace(go.Bar(
        x=['Total New Cost'],  # Single category for total new cost
        y=[inbound_cost],  # Inbound cost value
        name='Inbound Cost',
        marker_color='#EBCB8B',  # Yellow for inbound cost
        text=[inbound_cost],  # Add inbound cost label
        textposition='inside',  # Show text inside the bar
    ))

    # Add a stacked bar for Storage Handling
    fig.add_trace(go.Bar(
        x=['Total New Cost'],  # Same category for stacking
        y=[stor_handling],  # Storage handling cost value
        name='Storage/Handling Cost',
        marker_color='#A3BE8C',  # Green for storage handling
        text=[stor_handling],  # Add storage handling cost label
        textposition='inside',  # Show text inside the bar
    ))

    # Add a stacked bar for Outbound Cost
    fig.add_trace(go.Bar(
        x=['Total New Cost'],  # Same category for stacking
        y=[outbound_cost],  # Outbound cost value
        name='Outbound Cost',
        marker_color='#BF616A',  # Red for outbound cost
        text=[outbound_cost],  # Add outbound cost label
        textposition='inside',  # Show text inside the bar
    ))

    # Update layout for clear display
    fig.update_layout(
        barmode='stack',  # Stack bars for new cost components
        xaxis_title='Cost Components',  # X-axis title
        yaxis_title='Cost (€)',  # Y-axis title
        legend_title='Cost Breakdown',
        template='plotly_white',  # Clean look with white background
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),  # Remove margins around the chart
        height=400,  # Adjust height as needed
    )

    # Show the figure
    return fig
def split_(x):
        output = x.split('-')
        return output[1]

class Consolidation:
    def __init__(self, df, rate_df,dist_data, load_country, del_country, api_key):
        self.df = df
        self.rate_df = rate_df
        
        self.dist_data = dist_data[['Load_full_add', 'Delivery_full_add', 'Load_Coord', 'Del_Coord', 'Distance_km', 'Load_lat', 'Load_lon', 'Delivery_lat', 'Delivery_lon']].drop_duplicates()
        self.load_country = load_country
        self.del_country = del_country
        # self.gmaps = googlemaps.Client(key=api_key)
    
    def get_clean_data(self):
        df_clean = self.df.copy()
        df_clean['Load_full_add'] = df_clean['Load Address'] + "," + df_clean['Load City'] + "," + df_clean['Load Post Code'].astype('str') + "," + df_clean['Load Country']
        df_clean['Delivery_full_add'] = df_clean['Delivery Address'] + "," + df_clean['Delivery City'] + "," + df_clean['Delivery Post Code'].astype('str') + "," + df_clean['Delivery County']

        

        df_clean = df_clean.merge(self.dist_data, on=['Load_full_add', 'Delivery_full_add'], how='left')
        df_usecase1 = df_clean[(df_clean['Load Country'] == self.load_country) & (df_clean['Delivery County'].isin(self.del_country))]

        df_usecase1['Date'] = pd.to_datetime(df_usecase1['Actual Load Date'])

        useful_cols = ['Delivery Week','Tradeline','Order ID', 'Pallet Qty', 'Pallet Unit - Final', 'Temp Control?', 'Final Price', 'Load Point Name', 'Delivery Point Name', 'Load Country', 'Delivery County', 'Trade Lane',  'Load_full_add', 'Delivery_full_add', 'Distance_km', 'Load_lat', 'Load_lon', 'Delivery_lat', 'Delivery_lon','Delivery Lane']
        return df_usecase1[useful_cols]

    def calculate_distance(self, row, ib_lat, ib_lon):
        origin = (row['Load_lat'], row['Load_lon'])
        destination = (ib_lat, ib_lon)
        result = self.gmaps.distance_matrix(origins=[origin], destinations=[destination], mode='driving')
        distance_in_km = result['rows'][0]['elements'][0]['distance']['value'] / 1000  # Convert meters to km
        return distance_in_km

    def agg_data(self, df_usecase1, custom_location, location_address, ib_lat=0, ib_lon=0):
        agg_df = df_usecase1.groupby(['Delivery Week', 'Pallet Unit - Final', 'Temp Control?', 'Load_full_add', 'Delivery_full_add', 'Distance_km', 'Load_lat', 'Load_lon', 'Delivery_lat', 'Delivery_lon','Load Country','Delivery County','Delivery Lane']).agg({
            'Pallet Qty': 'sum', 'Final Price': 'sum', 'Order ID': pd.Series.nunique
        }).reset_index()

        ob_lat, ob_lon = agg_df['Delivery_lat'].values[0], agg_df['Delivery_lon'].values[0]

        if not custom_location:
            ib_lat = agg_df[agg_df['Load_full_add'] == location_address]['Load_lat'].values[0]
            ib_lon = agg_df[agg_df['Load_full_add'] == location_address]['Load_lon'].values[0]
            ob_dist = agg_df[agg_df['Load_full_add'] == location_address]['Distance_km'].values[0]
        else:
            #CURRENTLY NOT USING IT

            # ob_dist = self.gmaps.distance_matrix(origins=[(ib_lat, ib_lon)], destinations=[(ob_lat, ob_lon)], mode='driving')['rows'][0]['elements'][0]['distance']['value'] / 1000
            ob_dist = 101 #RANDOM
        
        agg_df['ib_lat'] = ib_lat
        agg_df['ib_lon'] = ib_lon
        agg_df['Distance_km_ob'] = [ob_dist for i in range(len(agg_df))]

        #CURRENTLY NOT USING IT
        # agg_df['Distance_km_ib'] = agg_df.apply(lambda row: self.calculate_distance(row, ib_lat, ib_lon), axis=1)
        agg_df['Distance_km_ib'] = [100 for i in range(len(agg_df))] #RANDOM 

        inbound_df = agg_df.groupby(['Load_full_add', 'Load_lat', 'Load_lon', 'ib_lat', 'ib_lon', 'Distance_km_ib']).sum().reset_index()
        storage_df = agg_df.groupby(['Delivery Week', 'Pallet Unit - Final', 'Temp Control?', 'Load_full_add', 'Load_lat', 'Load_lon', 'ib_lat', 'ib_lon']).sum().reset_index()
        outbound_df = agg_df.groupby(['Delivery Week', 'Pallet Unit - Final', 'Temp Control?', 'ib_lat', 'ib_lon', 'Delivery_full_add', 'Delivery_lat', 'Delivery_lon', 'Distance_km_ob','Delivery County','Load Country','Delivery Lane'])[['Pallet Qty','Final Price','Order ID']].sum().reset_index()
        #outbound_df = agg_df.groupby(['Delivery Week', 'Pallet Unit - Final', 'Temp Control?','ib_lat','ib_lon','Delivery_full_add','Delivery_lat','Delivery_lon','Distance_km_ob'])
        return inbound_df, storage_df, outbound_df, agg_df['Final Price'].sum()
    

    def avg_ship_rates(self, unit,custom_location,location_address,df_clean):
        rate_df = self.rate_df.copy()
        if unit == 'EUR':
            rate_df = rate_df[rate_df['Pallet Unit - Final'] == 'Eur']
        elif unit == 'IND':
            rate_df = rate_df[rate_df['Pallet Unit - Final'] == 'Ind']

        rate_df = rate_df.dropna(subset=['Tradeline'])
        rate_df['Delivery Lane'] = rate_df['Tradeline'].apply(lambda x: split_(x))
        rate_df = rate_df[['Load Country','Delivery Lane' ,'Delivery County', 'Orders (1-7)', 'Total Cost (1-7)']].dropna()

        
        try:
            ib_rate = rate_df[(rate_df['Load Country'] == self.load_country) & (rate_df['Delivery County'] == self.load_country)]
            per_order_ib_rate = ib_rate['Total Cost (1-7)'].sum() / ib_rate['Orders (1-7)'].sum()
        except:
            ib_rate = rate_df[(rate_df['Load Country'] == self.load_country)]
            per_order_ib_rate = ib_rate['Total Cost (1-7)'].sum() / ib_rate['Orders (1-7)'].sum()

        ob_rate = rate_df[(rate_df['Load Country'] == self.load_country)]
        # if not custom_location:
        #     tl = df_clean[df_clean['Load_full_add']==location_address]['Tradeline'].values[0]
  
        #     ob_rate = ob_rate[ob_rate['Tradeline']==tl]
        ob_rate_df = ob_rate.groupby(['Load Country','Delivery County','Delivery Lane'])[['Total Cost (1-7)','Orders (1-7)']].sum().reset_index()
        ob_rate_df['Rate'] = ob_rate_df['Total Cost (1-7)']/ob_rate_df['Orders (1-7)']
                        
        per_order_ob_rate = ob_rate['Total Cost (1-7)'].sum() / ob_rate['Orders (1-7)'].sum()

        return per_order_ib_rate, ob_rate_df[['Load Country','Delivery County','Delivery Lane','Rate']]

    def get_inbound_cost(self, inbound_df, per_order_ib_rate):
        inbound_df['cost_per_ship'] = per_order_ib_rate
        inbound_df['cost_per_ship'] = np.where(
            (inbound_df['Load_lat'] == inbound_df['ib_lat']) & (inbound_df['Load_lon'] == inbound_df['ib_lon']),
            0,
            inbound_df['cost_per_ship']
        )
        inbound_df['Inbound Cost'] = inbound_df['cost_per_ship'] * inbound_df['Order ID']
        return inbound_df['Inbound Cost'].sum()

    def get_storage_cost(self, storage_df, handling_cost, storage_cost):
        storage_df['Handling Cost'] = storage_df['Pallet Qty'] * handling_cost
        storage_df['Storage Cost'] = storage_df['Pallet Qty'] * storage_cost
        storage_df['Total Cost'] = storage_df['Handling Cost'] + storage_df['Storage Cost']
        storage_df['Total Cost'] = np.where(
            (storage_df['Load_lat'] == storage_df['ib_lat']) & (storage_df['Load_lon'] == storage_df['ib_lon']),
            0,
            storage_df['Total Cost']
        )
        return storage_df['Total Cost'].sum()

    def get_outbound_cost(self, outbound_df, per_order_ob_rate, max_qty):
        print(per_order_ob_rate)
        outbound_df['No of Trucks'] = np.ceil(outbound_df['Pallet Qty'] / max_qty)
        print(outbound_df['No of Trucks'].sum(),outbound_df['Pallet Qty'].sum())
        outbound_df = outbound_df.merge(per_order_ob_rate,on=['Load Country','Delivery Lane'],how='left')
        outbound_df['Outbound Cost'] = outbound_df['No of Trucks'] * outbound_df['Rate']
        outbound_df.to_csv("OB-CHECK1.csv")
        return outbound_df['Outbound Cost'].sum()

    def run_consolidation(self, custom_location, location_address,ib_lat,ib_lon,unit,handling_rate,storage_rate,max_qty):
        df_clean = self.get_clean_data()
        ib, sto, ob, existing_cost = self.agg_data(df_clean, custom_location, location_address,ib_lat,ib_lon)
        ib_rate, ob_rate = self.avg_ship_rates(unit,custom_location,location_address,df_clean)
        ib_cost = self.get_inbound_cost(ib, ib_rate)
        sto_cost = self.get_storage_cost(sto, handling_rate, storage_rate)
        ob_cost = self.get_outbound_cost(ob, ob_rate, max_qty)
        return int(ib_cost), int(sto_cost), int(ob_cost) , int(existing_cost)

def load_data():
    df = pd.read_excel('2025_Budget_Road.xlsx', sheet_name="Data")
    df.columns = df.iloc[7]
    df = df.iloc[8:]
    rate_df = pd.read_excel('2025_Budget_Road.xlsx', sheet_name="Budget - OPI")
    rate_df.columns = rate_df.iloc[5]
    rate_df = rate_df.iloc[6:]

    dist_data = pd.read_excel("Road Dashboard Data with Distance2.xlsx")

    return df,rate_df,dist_data

def main():

    icon = Image.open('logo1.png')
    image = Image.open("logo2.png")
    st.set_page_config(
    page_title="Green Field Simulator",
    page_icon=icon,  # Use your image file or provide a URL
    layout="wide",
    initial_sidebar_state="expanded",
        )   
    col7, col8 = st.columns([1, 4])  # Adjust the ratio as per your layout preference

    with col7:
        st.image(image, width=150)

    with col8:
        st.markdown("### Europe Network Cost Reduction - Sample Analysis")
    # Custom styles for compact sidebar and expander
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 300px !important; # Make sidebar compact
            }
            .streamlit-expanderHeader {
                background-color: #f0f0f0;
                color: #333; # Header color
            }
            .streamlit-expanderContent {
                background-color: #f9f9f9;
                color: #555; # Content color
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load data only once and store it in session state
    if 'data' not in st.session_state:
        with st.spinner("🌀 Loading data, please wait..."):
            df, rate_df, dist_data = load_data()
            st.session_state.data = {
                'df': df,
                'rate_df': rate_df,
                'dist_data': dist_data,
            }

    # Retrieve datasets from session state
    data = st.session_state.data
    df = data['df']
    rate_df = data['rate_df']
    dist_data = data['dist_data']

    
    # Header section with options
    # st.title("Supplier Consolidation Application")
    # st.markdown("Welcome to the Green Field Simulator")

    page = st.sidebar.selectbox("Select Page", ["Consolidator","Analysis"], index=0)

    
    
    if page == "Consolidator":

    # Sidebar selections
        st.sidebar.header("Consolidation Settings")

        # Select Loading Country
        load_country_arr = df['Load Country'].unique().tolist()
        load_country = st.sidebar.selectbox("Select Loading Country", load_country_arr)

        # Filter Delivery Country based on selected Load Country

        # del_country = 'Belgium'

        # Filter available consolidation locations based on selections
        
        ib_lat = 0.0  # Example latitude (can be replaced)
        ib_lon = 0.0  # Example longitude (can be replaced)
        location_address = 'NA'
        # Custom location selection logic

        # unit = st.sidebar.selectbox("Select Pallet Unit", ['EUR', 'IND', 'ALL'])
        unit = 'ALL'
        # Additional parameters
        custom_location_check = st.sidebar.checkbox("Use Custom Consolidation Point")
        if custom_location_check:
            trandeline_map = dist_data[(dist_data['Load Country'] == load_country)]
            # trandeline_map = trandeline_map[trandeline_map['Load Address'].isin(load_add)]
            map_df = trandeline_map[['Load_full_add', 'Delivery_full_add', 'Load_Coord', 'Del_Coord']].drop_duplicates()

            ib_lat,ib_lon = plot_interactive_map(map_df)
            st.sidebar.write(f"**Selected Lat:** {ib_lat}")
            st.sidebar.write(f"**Selected Lon:** {ib_lon}")
            custom_location = True
        else:
            loc_arr = dist_data[(dist_data['Load Country'] == load_country)]['Load_full_add'].unique().tolist()
            location_address = st.sidebar.selectbox("Select Consolidation Address", loc_arr)
            custom_location = False
        
        delivery_country_arr = df[df['Load Country'] == load_country]['Delivery County'].unique().tolist()
        del_country = st.sidebar.multiselect("Select Delivery Countries", ['ALL']+delivery_country_arr, default=['ALL'],help="Use the search bar to find countries")
        if 'ALL' in del_country or len(del_country)==0:
            del_country = list(delivery_country_arr)

        # del_country = []

        # # Create a checkbox for 'ALL' selection
        # select_all = st.sidebar.checkbox('Select All Delivery Countries', value=False)

        # # Checkbox grid for delivery countries
        # for country in delivery_country_arr:
        #     if select_all or st.sidebar.checkbox(country):
        #         del_country.append(country)

        # # Logic to handle selections
        # if select_all:
        #     del_country = delivery_country_arr  # Select all countries if 'ALL' is checked

    

        col1, col2, col3 = st.columns(3)


        with col1:
            handling_rate = st.number_input("Handling Rate", value=5.0)

        with col2:
            storage_rate = st.number_input("Storage Rate", value=2.5)

        with col3:
            max_qty = st.number_input("Truck Max Qty", value=66)
        
        
        # Consolidation calculations
        # api_key = "AIzaSyCdKK7F58zXrP4w1_aE-DeI6oxLbgvhAwI"
        api_key = 'NOT REQD'
        consolidator = Consolidation(df, rate_df, dist_data, load_country, del_country, api_key)


        ib_cost, sto_cost, ob_cost, existing_cost = consolidator.run_consolidation(custom_location, location_address, ib_lat, ib_lon, unit, handling_rate, storage_rate, max_qty)


        cost_data = {
            'Cost Type': ['Inbound Cost', 'Storage Cost', 'Outbound Cost','Existing Cost'],
            'Amount': [ib_cost, sto_cost, ob_cost,existing_cost]
        }
        cost_df = pd.DataFrame(cost_data)
        new_cost = ib_cost + ob_cost + sto_cost
        old_cost = existing_cost
        diff = old_cost - new_cost 

        if diff < 0:
            label = "Cost Increase"
            number = abs(round(diff,2))
            delta = "- " + str(round((number / old_cost) * 100, 2)) +"%"
            if number > 1000:
                number = str(number//1000) + "K"
            else:
                number = str(number)
        else:
            label = "Cost Savings"
            number = abs(round(diff,2))
            delta = "+ " + str(round((number / old_cost) * 100, 2)) +"%"
            if number > 1000:
                number = str(number//1000) + "K"
            else:
                number = str(number)
        
        st.metric(label=label, value="€ " +str(number), delta=delta)

        st.write("### Consolidation Costs Summary")
        a_col,b_col = st.columns([2,3])

        with b_col:
            fig = plot_result(ib_cost,sto_cost,ob_cost,existing_cost)
            config = {
            'displayModeBar': False,  # Remove the modebar (top tools like zoom, save)
            }
            st.plotly_chart(fig,config=config,use_container_width=True)

        with a_col:
            st.dataframe(cost_df.set_index(cost_df.columns[0]),width=300)
        


        #-------------INTERACTIVE MAP---------------
        
        # print(map_output)
        # print(lat,lon)
    
    if page == 'Analysis':
        # Sidebar setup
        st.sidebar.header("Analysis Page")

        # Utilization DataFrame calculation
        util_df = df.groupby(['Load Country', 'Delivery County', 'Trade Lane'])[['Pallet Qty', 'Truck Max ','Final Price']].sum().reset_index()
        util_df['Utilization %'] = (util_df['Pallet Qty'] / util_df['Truck Max ']) * 100
        util_df['Utilization %'] = pd.to_numeric(util_df['Utilization %']).round(2)
        util_df['Final Price'] = pd.to_numeric(util_df['Final Price']).round(2)
        util_df = util_df[['Load Country', 'Delivery County', 'Pallet Qty', 'Utilization %','Final Price']].rename(columns={'Final Price':'Existing Cost'})

        # Columns for layout
        
        cutoff = st.sidebar.number_input("Utilization Cutoff", value=75)

        # Column 1: Utilization Cutoff and Utilization Table
        
            
            
            # Filter DataFrame based on Utilization cutoff
            

        # Sidebar: Load Country and Delivery Country Filters
        st.sidebar.subheader("Filter by Country")
        
        # Select Loading Country
        load_country_arr = df['Load Country'].unique().tolist()
        load_country = st.sidebar.selectbox("Select Loading Country", load_country_arr)

        # Filter Delivery Country based on selected Load Country
        # delivery_country_arr = df[df['Load Country'] == load_country]['Delivery County'].unique().tolist()
        # del_country = st.sidebar.selectbox("Select Delivery Country", delivery_country_arr)

        # Dataframe filtered by selected countries
        trandeline = df[(df['Load Country'] == load_country)]
        df2 = trandeline.groupby(['Load Point Name','Load Address', 'Delivery Address', 'Company Name','Delivery County']).agg({'Order ID': pd.Series.nunique, 'Pallet Qty': 'sum','Truck Max ':'sum'}).reset_index().rename(columns={'Order ID':'# Shipments'}).reset_index(drop=True)
        df2['Utilization %'] = (df2['Pallet Qty'] / df2['Truck Max ']) * 100
        df2['Utilization %'] = pd.to_numeric(df2['Utilization %']).round(2)
        load_add = df2['Load Address'].unique().tolist()
        # Column 2: Filtered Delivery Data and Map
        
        filtered_util_df = util_df[util_df['Utilization %'] < cutoff]
        filtered_util_df = filtered_util_df.sort_values(by='Pallet Qty', ascending=False).iloc[:20].reset_index(drop=True)

        st.subheader("Top 20 High Volume Lanes (Post Cutoff) Utilization Overview")
        st.dataframe(filtered_util_df.set_index(filtered_util_df.columns[0]), height=300, width=900)
        col1, col2 = st.columns(2)  # Set the first column wider than the second one
        # with col1:
        st.subheader("Filtered Tradelines Data")
        # html_table = df2.to_html(index=False)
        
        data_ = df2[['Delivery County','Load Address','Delivery Address','Pallet Qty','# Shipments','Utilization %']]
        data_2 = df2[['Load Point Name','Company Name']]
        html_table = data_.to_html(index=False)
        # st.write(html_table, unsafe_allow_html=True)
        st.dataframe(data_.set_index(data_.columns[0]),width=800)
        #st.dataframe(data_)
        # Filter and sort utilization data


        # with col2:
        st.subheader("Tradelines")
        col_button,col_map = st.columns([1,5])

        with col_button:
            delivery_country_arr = df[df['Load Country'] == load_country]['Delivery County'].unique().tolist()
            del_country = st.selectbox("Select Delivery Country", ['ALL']+delivery_country_arr)
        with col_map:
            if del_country!='ALL':
                trandeline_map = dist_data[(dist_data['Load Country'] == load_country) & (dist_data['Delivery County'] == del_country)]
            else:
                trandeline_map = dist_data[(dist_data['Load Country'] == load_country)]
            trandeline_map = trandeline_map[trandeline_map['Load Address'].isin(load_add)]
            map_df = trandeline_map[['Load_full_add', 'Delivery_full_add', 'Load_Coord', 'Del_Coord']].drop_duplicates()

            # Plot the delivery map
            fig1 = plot_delivery_map(map_df)
            st.plotly_chart(fig1)
            
            # Display the second dataframe with a smaller width and height
            

        # Add some styling
        st.markdown("""
        <style>
            .css-1v3fvcr {  /* Streamlit container styling */
                padding: 10px;
            }
            .css-12oz5g7 {  /* Subheader styling */
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 15px;
            }
        </style>
        """, unsafe_allow_html=True)

            # Prepare data for the map


        # Add spacing between sections for better alignment
        st.markdown("<br><br>", unsafe_allow_html=True)



            


if __name__ == "__main__":
    main()       






