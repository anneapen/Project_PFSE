import streamlit as st
import Project.wind_loads as wind_loads
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



st.sidebar.header("**StructWind**")
st.header("""**Wind Loads as per IS 875: Part-3:2015**""",)

location_list=['Agra', 'Ahmedabad', 'Ajmer', 'Almora', 'Amritsar', 'Asansol', 'Aurangabad', 
            'Bahraich', 'Barauni', 'Bengaluru', 'Bareilly', 'Bhatinda', 'Bhilai', 'Bhopal',
            'Bhubaneshwar', 'Bhuj', 'Bikaner', 'Bokaro', 'Chandigarh', 'Chennai', 'Coimbatore', 
            'Cuttack', 'Darbhanga', 'Darjeeling', 'Dehradun', 'Delhi', 'Durgapur', 'Gangtok', 
            'Guwahati', 'Gaya', 'Gorakhpur', 'Hyderabad', 'Imphal', 'Jabalpur', 'Jaipur', 'Jamshedpur',
            'Jhansi', 'Jodhpur', 'Kanpur', 'Kohima', 'Kolkata', 'Kozhikode', 'Kurnool', 'Lakshadeep', 
            'Lucknow', 'Ludhiana', 'Madurai', 'Mandi', 'Mangalore', 'Moradabad', 'Mumbai', 'Mysore',
            'Nagpur', 'Nainital', 'Nasik', 'Nellore', 'Panjim', 'Patiala', 'Patna', 'Puducherry', 
            'Port Blaire', 'Pune', 'Raipur', 'Rajkot', 'Ranchi', 'Rourkee', 'Rourkela', 'Shimla', 
            'Srinagar', 'Surat', 'Tiruchirapalli', 'Trivandrum', 'Udaipur', 'Vadodara', 'Varanasi', 
            'Vijayawada', 'Vishakapatanam']

structural_class_list=['General','Temporary structures','Structures with low risk','Important structures']
terrain_category_list=['Category 1','Category 2','Category 3','Category 4']
str_imp_list=['Important Structures','Industrial Structures','Other Structures']
str_form_list=['Triangular','Square','Rectangular','Circular','Cyclone affected region']
str_kind_list=['Welded steel structures','Bolted steel structures/RCC structures','Prestressed Concrete Structures']

location=st.sidebar.selectbox("Location",options=location_list)
height=st.sidebar.number_input("Height(m)",min_value=0.0,value=10.0)
storey_ht=st.sidebar.number_input("Average Storey height(m)",min_value=0.0,value=3.0)
structural_class=st.sidebar.selectbox("Class of structure",options=structural_class_list)
category=st.sidebar.selectbox("Terrain Category",options=terrain_category_list)
slope=st.sidebar.number_input("Slope(in deg)",min_value=0.0,step=0.1)
structure_imp=st.sidebar.selectbox("Importance of structure",options=str_imp_list)
structural_form=st.sidebar.selectbox("Structural form",options=str_form_list)
tributary_area=st.sidebar.number_input("Tributary area(sq.m)",min_value=0)
Cpe=st.sidebar.number_input("External pressure coefficient (Cpe)",min_value=-1.0, max_value=1.0,step=0.01)
Cpi=st.sidebar.number_input("Internal pressure coefficient (Cpi)",min_value=-1.0, max_value=1.0,step=0.01)
area=st.sidebar.number_input("Effective frontal area(sqm)",min_value=0)
Cfx=st.sidebar.number_input("Force coefficient in X-dirn (Cfx)",min_value=0.0, max_value=3.0,step=0.01,value=1.2)
Cfy=st.sidebar.number_input("Force coefficient in Y-dirn (Cfy)",min_value=0.0, max_value=3.0,step=0.01,value=1.2)
structural_kind=st.sidebar.selectbox("Kind of structure",options=str_kind_list)


vb,k1=wind_loads.basic_wind_speed(location,structural_class)
k2=wind_loads.terrain_factor(height,category)
k4=wind_loads.importance_factor(structure_imp)
Kd=wind_loads.wind_directionality_factor(structural_form)
Ka=wind_loads.area_averaging_factor(tributary_area)
k3, Vz, pz, Kc, p_design=wind_loads.design_wind_pressure(location,structural_class,height,category,slope,
                         structure_imp,structural_form,tributary_area)
F=wind_loads.load_calc_pressure(Cpe,Cpi,area,p_design)
beta=wind_loads.damping_coefficient(structural_kind)

tab1, tab2 = st.tabs(["General Design", "Multistorey building"])

with tab1:
   st.subheader("Wind Pressure and Forces")
   st.write(f"Basic Wind speed,Vb={vb}m/s")
   st.write(f"Probability factor(Risk coefficient),k1={k1}")
   st.write(f"Terrain roughness and height factor,k2={k2}")
   st.write(f"Topography factor,k3={k3}")
   st.write(f"Importance factor,k4={k4}")
   st.markdown(f"***Design Wind speed,Vz=Vbxk1xk2xk3xk4={Vz}m/s***")
   st.write(f"Wind pressure,pz={pz}m/s")
   st.write(f"Wind directionality factor,Kd={Kd}")
   st.write(f"Area averaging factor,Ka={Ka}")
   st.write(f"Combination factor,Kc={Kc}")
   st.markdown(f"***Design wind pressure,pd=KdxKaxKcxpz={p_design}kN/m2***")
   st.markdown(f"***Design wind load,F=(Cpe-Cpi)xareaxpd={F}kN***")    


with tab2:
   st.subheader("Wind Pressure and Forces on Multi-storey building") 
   st.markdown("***Check for Dynamic Effects***") 

   lx=st.number_input("Plan Dimension in X-dirn(m)",min_value=0.0,value=10.0)
   ly=st.number_input("Plan Dimension in Y-dirn(m)",min_value=0.0,value=5.0)
   Tx=st.number_input("Time period in X-dirn(s)",min_value=0.0,step=0.01,value=2.0)
   Ty=st.number_input("Time period in Y-dirn(s)",min_value=0.0,step=0.01,value=2.0)

   if ((height/min(lx,ly))>5 or (1/Tx)<1 or (1/Tx)<1):
      st.markdown("***Dynamic analysis required***")
      if st.button("Dynamic Analysis"):
         st.write("***Wind Pressure and Forces on Multi-storey building (Gust Factor Approach)***")
         # Initialize the DataFrame in session state if not already present
         if "data_df" not in st.session_state:
            st.session_state.data_df = []

         # Function to add a new row
         def add_storey():
            # Append new data and update the DataFrame in session state
            st.session_state.data_df.append({"Storey": st.session_state.Storey, "Height": st.session_state.Height})
            # Clear the input fields by resetting them
            st.session_state.Storey = ""
            st.session_state.Height = ""

         # Function to clear all entries
         def clear_entries():
            # Clear the DataFrame in session state
            st.session_state.data_df = []

         # Use a form to hold inputs and buttons
         with st.form("entry_form"):
            # Collect inputs; use session_state to hold current inputs
            storey = st.text_input("Storey level(from top)", key="Storey", value=st.session_state.get('Storey', ''))
            storey_height = st.text_input("Storey Height(m)", key="Height", value=st.session_state.get('Height', ''))
            # Create form buttons for adding and clearing entries
            submitted = st.form_submit_button("Add Storey", on_click=add_storey)
            cleared = st.form_submit_button("Clear Entries", on_click=clear_entries)

         # Display the DataFrame if it exists
         if "data_df" in st.session_state and st.session_state.data_df:
            data_df = pd.DataFrame(st.session_state.data_df)
            st.table(data_df)
            
            data_df.iloc[:, 1] = data_df.iloc[:, 1].astype(float)
            storey_data_df=data_df.iloc[:, 1]
            if isinstance(storey_data_df, pd.Series):
               storey_data_df = storey_data_df.to_frame()
            
               try:
                  dyn_pressure_df = wind_loads.dynamic_wind_pressure(storey_data_df,height,category,vb,k1,k3,k4,lx,ly,Tx,Ty,beta,Cfx,Cfy)
                  st.write(dyn_pressure_df)                                  
               except Exception as e:
                  st.error(f"An error occurred: {str(e)}")        
                   
              
               
           
      if st.button("Static Analysis"):    
         st.write("***Wind Pressure and Forces on Multi-storey building (Static Analysis)***")
         st.write("*Calculations are based on given height,average storey height and effective frontal area")
         st.write(f"Basic Wind speed,Vb={vb}m/s")
         st.write(f"Probability factor(Risk coefficient),k1={k1}")
         st.write(f"Topography factor,k3={k3}")
         st.write(f"Importance factor,k4={k4}")
         st.write(f"Wind directionality factor,Kd={Kd}")
         st.write(f"Area averaging factor,Ka={Ka}")
         st.write(f"Combination factor,Kc={Kc}")
         st.markdown("***Design wind pressure,pd=KdxKaxKcxpz***")
         st.markdown("***Design wind load,F=Cfxareaxpd***")
         st.markdown("***Variation in design wind speed, design pressure and loads with height***")
         df=wind_loads.wind_data_frame(height,storey_ht,category,vb,k1,k3,k4,Kd,Ka,Kc,Cfx,area)
         st.table(df)   

         degree = 5  # Degree of the polynomial to fit
         coefficients = np.polyfit(df['Vz(m/s)'], df['Height'], degree)
         polynomial = np.poly1d(coefficients)

         # Generate new x values to create a smooth curve
         Vz_smooth = np.linspace(df['Vz(m/s)'].min(), df['Vz(m/s)'].max(), 500)
         Height_smooth = polynomial(Vz_smooth)

         plt.figure(figsize=(8, 6))
         plt.scatter(df['Vz(m/s)'], df['Height'])  # Plot original points
         plt.plot(Vz_smooth, Height_smooth, 'blue')  # Plot smooth curve
         plt.xlabel('Vz(m/s)')
         plt.ylabel('Height')
         plt.title('Variation in Design Wind Speed with Height')
         plt.ylim(10, None)
         plt.grid(True)
         plt.tight_layout()
         plt.show()

         # Display the plot in Streamlit
         st.pyplot(plt)
         

   

   else:
      st.markdown("***Dynamic analysis not required***")
      if st.button("Static Analysis"):    
         st.write("***Wind Pressure and Forces on Multi-storey building (Static Analysis)***")
         st.write("*Calculations are based on given height,average storey height and effective frontal area")            
         st.write(f"Basic Wind speed,Vb={vb}m/s")
         st.write(f"Probability factor(Risk coefficient),k1={k1}")
         st.write(f"Topography factor,k3={k3}")
         st.write(f"Importance factor,k4={k4}")
         st.write(f"Wind directionality factor,Kd={Kd}")
         st.write(f"Area averaging factor,Ka={Ka}")
         st.write(f"Combination factor,Kc={Kc}")
         st.markdown("***Design wind pressure,pd=KdxKaxKcxpz***")
         st.markdown("***Design wind load,F=Cfxareaxpd***")
         st.markdown("***Variation in design wind speed, design pressure and loads with height***")
         df=wind_loads.wind_data_frame(height,storey_ht,category,vb,k1,k3,k4,Kd,Ka,Kc,Cfx,area)
         st.table(df)  
         degree = 5  # Degree of the polynomial to fit
         coefficients = np.polyfit(df['Vz(m/s)'], df['Height'], degree)
         polynomial = np.poly1d(coefficients)

         # Generate new x values to create a smooth curve
         Vz_smooth = np.linspace(df['Vz(m/s)'].min(), df['Vz(m/s)'].max(), 500)
         Height_smooth = polynomial(Vz_smooth)

         plt.figure(figsize=(8, 6))
         plt.scatter(df['Vz(m/s)'], df['Height'])  # Plot original points
         plt.plot(Vz_smooth, Height_smooth, 'blue')  # Plot smooth curve
         plt.xlabel('Vz(m/s)')
         plt.ylabel('Height')
         plt.title('Variation in Design Wind Speed with Height')
         plt.ylim(10, None)
         plt.grid(True)
         plt.tight_layout()
         plt.show()

         # Display the plot in Streamlit
         st.pyplot(plt) 


   


