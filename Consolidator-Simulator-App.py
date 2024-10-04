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
    lat=lon=0.0
    
    center_coordinates = [54.5260, 15.2551]  # Initial view centered on San Francisco
    zoom_level = 4
    m = folium.Map(location=center_coordinates, zoom_start=zoom_level)

    # Add markers for Load and Delivery points
    for i, row in df.iterrows():
        load_lat, load_lon = str_to_tuple(row['Load_Coord'])
        del_lat, del_lon = str_to_tuple(row['Del_Coord'])
        
        # Add Load marker
        folium.Marker(
            location=[load_lat, load_lon],
            popup=f"Load Location {i+1}: {row['Load_Coord']}",
            icon=folium.Icon(color="green")
        ).add_to(m)
        
        # Add Delivery marker
        folium.Marker(
            location=[del_lat, del_lon],
            popup=f"Delivery Location {i+1}: {row['Del_Coord']}",
            icon=folium.Icon(color="red")
        ).add_to(m)

    # Add the LatLngPopup to capture clicks and return lat/lon
    # m.add_child(folium.LatLngPopup())

    # Display the map using Streamlit
    map_data = st_folium(m, width=700, height=500)

    # Capture the clicked location
    if map_data and map_data['last_clicked']:
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        # st.write(f"Clicked Latitude: {lat}")
        # st.write(f"Clicked Longitude: {lon}")
    else:
        st.write("Click on the map to get latitude and longitude.")
    
    return lat,lon


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

def plot_result(inbound_cost,stor_handling,outbound_cost,existing_cost):
    new_cost_total = inbound_cost + stor_handling + outbound_cost

    # Calculate the difference in values and percentages
    value_diff = existing_cost - new_cost_total
    percent_diff = (value_diff / existing_cost) * 100

    # Create the figure
    fig = go.Figure()

    # Bar for existing cost
    fig.add_trace(go.Bar(
        x=['Existing Cost'],
        y=[existing_cost],
        name='Existing Cost',
        marker_color='lightblue',
        text=[existing_cost],  # Add data labels
        textposition='auto'  # Show text inside or on top of bars
    ))

    # Stacked bar for new cost components
    fig.add_trace(go.Bar(
        x=['Optimized Cost'],
        y=[new_cost_total],
        name='Cost After Consolidation',
        marker_color='lightgreen',
        text=[new_cost_total],  # Add data labels
        textposition='auto'  # Show text inside or on top of bars
    ))




    # Add annotation for the difference between costs
    fig.add_annotation(
        x=1,  # Position between the two bars
        y=new_cost_total + new_cost_total*0.2,  # Slightly above the higher bar
        text=f"Difference: €{value_diff:,} ({percent_diff:.2f}%)",
        showarrow=False,
        font=dict(size=12, color="black"),
        align='center'
    )

    # Update layout for stacked bars
    fig.update_layout(
        barmode='stack',  # Stacks bars for new cost
        # title='Existing vs. New Cost Comparison',
        xaxis_title='Cost Type',
        yaxis_title='Total Cost (€)',
        legend_title='Cost Breakdown',
        template='plotly_white',  # Clean look
        showlegend=True,
        width = 800
    )

    # Show the figure
    return fig


class Consolidation:
    def __init__(self, df, rate_df,dist_data, load_country, del_country, api_key):
        self.df = df
        self.rate_df = rate_df
        
        self.dist_data = dist_data[['Load_full_add', 'Delivery_full_add', 'Load_Coord', 'Del_Coord', 'Distance_km', 'Load_lat', 'Load_lon', 'Delivery_lat', 'Delivery_lon']].drop_duplicates()
        self.load_country = load_country
        self.del_country = del_country
        self.gmaps = googlemaps.Client(key=api_key)
    
    def get_clean_data(self):
        df_clean = self.df.copy()
        df_clean['Load_full_add'] = df_clean['Load Address'] + "," + df_clean['Load City'] + "," + df_clean['Load Post Code'].astype('str') + "," + df_clean['Load Country']
        df_clean['Delivery_full_add'] = df_clean['Delivery Address'] + "," + df_clean['Delivery City'] + "," + df_clean['Delivery Post Code'].astype('str') + "," + df_clean['Delivery County']

        

        df_clean = df_clean.merge(self.dist_data, on=['Load_full_add', 'Delivery_full_add'], how='left')
        df_usecase1 = df_clean[(df_clean['Load Country'] == self.load_country) & (df_clean['Delivery County'] == self.del_country)]
        df_usecase1['Date'] = pd.to_datetime(df_usecase1['Actual Load Date'])

        useful_cols = ['Delivery Week','Tradeline','Order ID', 'Pallet Qty', 'Pallet Unit - Final', 'Temp Control?', 'Final Price', 'Load Point Name', 'Delivery Point Name', 'Load Country', 'Delivery County', 'Trade Lane',  'Load_full_add', 'Delivery_full_add', 'Distance_km', 'Load_lat', 'Load_lon', 'Delivery_lat', 'Delivery_lon']
        return df_usecase1[useful_cols]

    def calculate_distance(self, row, ib_lat, ib_lon):
        origin = (row['Load_lat'], row['Load_lon'])
        destination = (ib_lat, ib_lon)
        result = self.gmaps.distance_matrix(origins=[origin], destinations=[destination], mode='driving')
        distance_in_km = result['rows'][0]['elements'][0]['distance']['value'] / 1000  # Convert meters to km
        return distance_in_km

    def agg_data(self, df_usecase1, custom_location, location_address, ib_lat=0, ib_lon=0):
        agg_df = df_usecase1.groupby(['Delivery Week', 'Pallet Unit - Final', 'Temp Control?', 'Load_full_add', 'Delivery_full_add', 'Distance_km', 'Load_lat', 'Load_lon', 'Delivery_lat', 'Delivery_lon']).agg({
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
        outbound_df = agg_df.groupby(['Delivery Week', 'Pallet Unit - Final', 'Temp Control?', 'ib_lat', 'ib_lon', 'Delivery_full_add', 'Delivery_lat', 'Delivery_lon', 'Distance_km_ob'])[['Pallet Qty','Final Price','Order ID']].sum().reset_index()
        #outbound_df = agg_df.groupby(['Delivery Week', 'Pallet Unit - Final', 'Temp Control?','ib_lat','ib_lon','Delivery_full_add','Delivery_lat','Delivery_lon','Distance_km_ob'])
        return inbound_df, storage_df, outbound_df, agg_df['Final Price'].sum()

    def avg_ship_rates(self, unit,custom_location,location_address,df_clean):
        rate_df = self.rate_df.copy()
        if unit == 'EUR':
            rate_df = rate_df[rate_df['Pallet Unit - Final'] == 'Eur']
        elif unit == 'IND':
            rate_df = rate_df[rate_df['Pallet Unit - Final'] == 'Ind']
        rate_df = rate_df[['Load Country','Tradeline' ,'Delivery County', 'Orders (1-7)', 'Total Cost (1-7)']].dropna()

        
        try:
            ib_rate = rate_df[(rate_df['Load Country'] == self.load_country) & (rate_df['Delivery County'] == self.load_country)]
            per_order_ib_rate = ib_rate['Total Cost (1-7)'].sum() / ib_rate['Orders (1-7)'].sum()
        except:
            ib_rate = rate_df[(rate_df['Load Country'] == self.load_country)]
            per_order_ib_rate = ib_rate['Total Cost (1-7)'].sum() / ib_rate['Orders (1-7)'].sum()

        ob_rate = rate_df[(rate_df['Load Country'] == self.load_country) & (rate_df['Delivery County'] == self.del_country)]
        if not custom_location:
            tl = df_clean[df_clean['Load_full_add']==location_address]['Tradeline'].values[0]
  
            ob_rate = ob_rate[ob_rate['Tradeline']==tl]
        per_order_ob_rate = ob_rate['Total Cost (1-7)'].sum() / ob_rate['Orders (1-7)'].sum()

        return per_order_ib_rate, per_order_ob_rate

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
        outbound_df['Outbound Cost'] = outbound_df['No of Trucks'] * per_order_ob_rate
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
        delivery_country_arr = df[df['Load Country'] == load_country]['Delivery County'].unique().tolist()
        del_country = st.sidebar.selectbox("Select Delivery Country", delivery_country_arr)

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
            trandeline_map = dist_data[(dist_data['Load Country'] == load_country) & (dist_data['Delivery County'] == del_country)]
            # trandeline_map = trandeline_map[trandeline_map['Load Address'].isin(load_add)]
            map_df = trandeline_map[['Load_full_add', 'Delivery_full_add', 'Load_Coord', 'Del_Coord']].drop_duplicates()

            ib_lat,ib_lon = plot_interactive_map(map_df)
            st.sidebar.write(f"**Selected Lat:** {ib_lat}")
            st.sidebar.write(f"**Selected Lon:** {ib_lon}")
            custom_location = True
        else:
            loc_arr = dist_data[(dist_data['Load Country'] == load_country) & (dist_data['Delivery County'] == del_country)]['Load_full_add'].unique().tolist()
            location_address = st.sidebar.selectbox("Select Consolidation Address", loc_arr)
            custom_location = False

        col1, col2, col3 = st.columns(3)


        with col1:
            handling_rate = st.number_input("Handling Rate", value=5.0)

        with col2:
            storage_rate = st.number_input("Storage Rate", value=2.5)

        with col3:
            max_qty = st.number_input("Truck Max Qty", value=66)
        
        
        # Consolidation calculations
        api_key = "AIzaSyCdKK7F58zXrP4w1_aE-DeI6oxLbgvhAwI"
        consolidator = Consolidation(df, rate_df, dist_data, load_country, del_country, api_key)


        ib_cost, sto_cost, ob_cost, existing_cost = consolidator.run_consolidation(custom_location, location_address, ib_lat, ib_lon, unit, handling_rate, storage_rate, max_qty)


        cost_data = {
            'Cost Type': ['Inbound Cost', 'Storage Cost', 'Outbound Cost','Existing Cost'],
            'Amount': [ib_cost, sto_cost, ob_cost,existing_cost]
        }
        cost_df = pd.DataFrame(cost_data)


        st.write("### Consolidation Costs Summary")


        fig = plot_result(ib_cost,sto_cost,ob_cost,existing_cost)

        st.dataframe(cost_df)
            


        st.plotly_chart(fig)
        #-------------INTERACTIVE MAP---------------
        
        # print(map_output)
        # print(lat,lon)
    
    if page == 'Analysis':
        # Sidebar setup
        st.sidebar.header("Analysis Page")

        # Utilization DataFrame calculation
        util_df = df.groupby(['Load Country', 'Delivery County', 'Trade Lane'])[['Pallet Qty', 'Truck Max ']].sum().reset_index()
        util_df['Utilization %'] = (util_df['Pallet Qty'] / util_df['Truck Max ']) * 100
        util_df['Utilization %'] = pd.to_numeric(util_df['Utilization %']).round(2)
        util_df = util_df[['Load Country', 'Delivery County', 'Pallet Qty', 'Utilization %']].round(2)

        # Columns for layout
        col1, col2 = st.columns(2)
        cutoff = st.sidebar.number_input("Utilization Cutoff", value=75)

        # Column 1: Utilization Cutoff and Utilization Table
        with col1:
            st.subheader("Utilization Overview")
            
            
            # Filter DataFrame based on Utilization cutoff
            filtered_util_df = util_df[util_df['Utilization %'] < cutoff]
            filtered_util_df = filtered_util_df.sort_values(by='Pallet Qty', ascending=False).iloc[:20].reset_index(drop=True)

            st.dataframe(filtered_util_df, height=300, width=800)

        # Sidebar: Load Country and Delivery Country Filters
        st.sidebar.subheader("Filter by Country")
        
        # Select Loading Country
        load_country_arr = df['Load Country'].unique().tolist()
        load_country = st.sidebar.selectbox("Select Loading Country", load_country_arr)

        # Filter Delivery Country based on selected Load Country
        delivery_country_arr = df[df['Load Country'] == load_country]['Delivery County'].unique().tolist()
        del_country = st.sidebar.selectbox("Select Delivery Country", delivery_country_arr)

        # Dataframe filtered by selected countries
        trandeline = df[(df['Load Country'] == load_country) & (df['Delivery County'] == del_country)]
        df2 = trandeline.groupby(['Load Point Name','Load Address', 'Delivery Address', 'Company Name']).agg({'Order ID': pd.Series.nunique, 'Pallet Qty': 'sum'}).reset_index().rename(columns={'Order ID':'# Shipments'}).reset_index(drop=True)
        load_add = df2['Load Address'].unique().tolist()
        # Column 2: Filtered Delivery Data and Map
        with col2:
            st.subheader("Filtered Delivery Data")
            st.dataframe(df2)

            # Prepare data for the map
        trandeline_map = dist_data[(dist_data['Load Country'] == load_country) & (dist_data['Delivery County'] == del_country)]
        trandeline_map = trandeline_map[trandeline_map['Load Address'].isin(load_add)]
        map_df = trandeline_map[['Load_full_add', 'Delivery_full_add', 'Load_Coord', 'Del_Coord']].drop_duplicates()

            # Plot the delivery map
        fig1 = plot_delivery_map(map_df)
        st.plotly_chart(fig1)

        # Add spacing between sections for better alignment
        st.markdown("<br><br>", unsafe_allow_html=True)



            


if __name__ == "__main__":
    main()       






