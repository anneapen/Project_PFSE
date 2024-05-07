import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt


#Basic wind speed(Vb) and risk coefficient or probability factor(k1)
def basic_wind_speed(location:str,structural_class:str)->tuple[float,float]:
    """
    Returns the basic wind speed(vb) in m/s and risk coefficient(k1) for the provided location and
    class of structure as per IS 875 Part-3:2015
    """
    vb_df=pd.read_csv("Basic wind speed.csv")
    vb_df=vb_df.set_index("Location")
    k1_df=pd.read_csv("Risk coefficient factor.csv")
    k1_df=k1_df[:4]
    k1_df=k1_df.set_index("Structural class")
    location_list = vb_df.index.tolist()
    if (location in location_list):
        vb=vb_df.loc[location,'Vb']
        k1=k1_df.loc[structural_class,str(vb)]

    return vb,k1

#Terrain roughness and height factor(k2)
def terrain_factor(height:float,category:str)->float:
    """
    Returns the terrain roughness and height factor(k2) for the given height(m) and 
      category level as per IS 875 Part-3:2015
    """
    k2_df=pd.read_csv("Terrain factor.csv")
    k2_df=k2_df[:15]
    # Adding given height with a NaN value for interpolation, with data types specified
    new_data = pd.DataFrame({'Height': height, category: [None]}, dtype=float)
    k2_df = pd.concat([ k2_df, new_data], ignore_index=True)
    # Sorting the DataFrame by Height to ensure proper interpolation
    k2_df.sort_values('Height', inplace=True)
    # Performing linear interpolation
    k2_df[category] = k2_df[category].interpolate(method='linear')
    # Extracting the interpolated value for given height 
    k2 = k2_df[k2_df['Height'] == height][category].iloc[0]
    return round(k2,3)

#Importance factor(k4)
def importance_factor(structure_imp:str):
    """
    Returns the importance_factor(k4) for the given level of importance as per IS 875 Part-3:2015
    """
    k4_df=pd.read_csv("Importance factor.csv")
    k4_df=k4_df[:3]
    k4_df=k4_df.set_index("Structure")
    k4=k4_df.loc[structure_imp,'k4']
    return k4

#Area averaging factor(Ka)
def area_averaging_factor(tributary_area:float)->float:
    """
    Returns the area_averaging_factor(Ka) for the given tributary area(sqm) as per IS 875 Part-3:2015
    """
    ka_df=pd.read_csv("Area averaging factor.csv")
    ka_df=ka_df[:4]
    # Adding given tributary area with a NaN value for interpolation, with data types specified
    new_data = pd.DataFrame({'Tributary Area': tributary_area, 'Ka': [None]}, dtype=float)
    ka_df = pd.concat([ ka_df, new_data], ignore_index=True)
    # Sorting the DataFrame by tributary area to ensure proper interpolation
    ka_df.sort_values('Tributary Area', inplace=True)
    # Performing linear interpolation
    ka_df['Ka'] = ka_df['Ka'].interpolate(method='linear')
    # Extracting the interpolated value for given tributary area
    ka = ka_df[ka_df['Tributary Area'] == tributary_area]['Ka'].iloc[0]
    return round(ka,3)

#Wind directionality factor(Kd)
def wind_directionality_factor(structural_form:str):
    """
    Returns the wind_directionality_factor(Kd) as per IS 875 Part-3:2015
    """
    kd_df=pd.read_csv("Wind directionality factor.csv")
    kd_df=kd_df[:5]
    kd_df=kd_df.set_index("Structural form")
    kd=kd_df.loc[structural_form,'Kd']
    return kd

#Design wind pressure(pd)
def design_wind_pressure(location:str,structural_class:str,height:float,category:str,slope:float,
                         structure_imp:str,structural_form:str,tributary_area:float,
                        )->tuple[float,float,float,float,float,]:
    """
    Calculates the design_wind_pressure,pd(kN/m2) as per IS 875 Part-3:2015
        pd=Kd*Ka*Kc*pz
        pz=0.6*(Vz**2)
        Vz=vb*k1*k2*k3*k4
    where,
    location - city where the building is considered
    structural_class- class of structure for finding k1
    height-total height of the building in m 
    category - terrain category for obtaining k2
    slope- upwind slope to find the topography factor k3
    structure_imp- importance of the structure to find the importance factor k4
    structural_form- form of the structure to find the wind directionality factor Kd
    tributary_area -tributary area in m2 to find the avea averaging factor Ka
    Kc- combination factor

    """
    vb,k1=basic_wind_speed(location,structural_class)
    k2=terrain_factor(height,category)
    if (slope<3):
        k3=1.0
    else:
        k3=1.36
    k4=importance_factor(structure_imp)
    Vz=vb*k1*k2*k3*k4
    pz=0.6*(Vz**2)*(10**-3)
    Kd=wind_directionality_factor(structural_form)
    Ka=area_averaging_factor(tributary_area)
    Kc=0.9
    p_des=Kd*Ka*Kc*pz*10**-3
    if (p_des<=(0.7*pz)):
        pd=0.7*pz
    else:
        pd=p_des
    return k3,round(Vz,2),round(pz,2),Kc,round(pd,2)

#Wind load(F)  
def load_calc_pressure(Cpe:float,Cpi:float,area:float,pd:float)->float:
    """
    Calculates the wind load,F for the given external and internal pressure coefficients and 
    design wind pressure as per IS 875 Part-3:2015
    """
    
    F=(Cpe-Cpi)*area*pd
    return round(F,2)

def load_calc_force(Cf:float,area:float,pd:float)->float:
    """
    Calculates the wind load,F for the given force coefficients and 
    design wind pressure as per IS 875 Part-3:2015
    """
    
    F=Cf*area*pd
    return round(F,2)

def damping_coefficient(structural_kind:str):
    """
    Returns the damping_coefficient for the given kind of structure as per IS 875 Part-3:2015
    """
    beta_df=pd.read_csv("Kind of structure.csv")
    beta_df=beta_df.set_index("Kind of structure")
    beta=beta_df.loc[structural_kind,'beta']
    return beta

def turbulence_intensity(category: str, z: pd.Series) -> pd.Series:
    """
    Returns the turbulence intensity according to the given terrain category 
    as per IS 875 Part-3:2015.
    
    Parameters:
    - category : Terrain category.
    - z : Series of heights at which turbulence intensity to be calculated.

    """
    # Use numpy's log10 for vectorized operations
    Iz1 = 0.3507 - (0.0535 * np.log10(z / 0.002))
    Iz4 = 0.466 - (0.1358 * np.log10(z / 2))
    Iz3 = Iz1 + ((3 / 7) * (Iz4 - Iz1))
    Iz2 = Iz1 + ((1 / 7) * (Iz4 - Iz1))
    
    # Choose the correct Iz based on category
    if category == 'Category 1':
        Iz = Iz1
    elif category == 'Category 2':
        Iz = Iz2
    elif category == 'Category 3':
        Iz = Iz3
    else:
        Iz = Iz4
    
    return np.round(Iz, 3)

def terrain_factor_hourly(height:float,category:str)->float:
    """
    Returns the hourly terrain roughness and height factor(k2) for the given height(m) and 
      category level based on the equation given in IS 875 Part-3:2015 
      
    """
    hourlyk2_df=pd.read_csv("Hourly mean wind speed factor k2.csv")
    # Adding given height with a NaN value for interpolation, with data types specified
    new_data = pd.DataFrame({'Height': height, category: [None]}, dtype=float)
    hourlyk2_df = pd.concat([ hourlyk2_df, new_data], ignore_index=True)
    # Sorting the DataFrame by Height to ensure proper interpolation
    hourlyk2_df.sort_values('Height', inplace=True)
    # Performing linear interpolation
    hourlyk2_df[category] = hourlyk2_df[category].interpolate(method='linear')
    # Extracting the interpolated value for given height 
    hourlyk2 = hourlyk2_df[hourlyk2_df['Height'] == height][category].iloc[0]
    return round(hourlyk2,3)

def wind_data_frame(height:float,storey_height:float,category:str,Vb:float,k1:float,k3:float,k4:float,
                    Kd:float,Ka:float,Kc:float,Cf:float,area:float)->pd.DataFrame:
    """
    Creates a dataframe which shows the variation in design wind speed(Vz) and design pressure(pd) with height
    as per IS 875 Part-3:2015
    """
    
    k2_df=pd.read_csv("Terrain factor.csv")
    k2_df=k2_df[:15]
    heights = np.arange(0, height + 1, storey_height)  
    if heights[-1] != height:
        heights = np.append(heights, height)

    # Creating a new dataframe to hold interpolated values
    new_df = pd.DataFrame({'Height': heights})

    # Interpolating values for given category
    new_df[category] = np.round(np.interp(heights, k2_df['Height'], k2_df[category]),2)
    new_df.rename(columns={category: 'k2'}, inplace=True)
    new_df['Vz(m/s)'] = round(Vb * k1 * new_df['k2'] * k3 * k4,2)
    new_df['pz(kN/m2)'] = round((0.6 * new_df['Vz(m/s)'] * new_df['Vz(m/s)']*10**-3),3)
    new_df['pd(kN/m2)'] =round(Kd*Ka*Kc*new_df['pz(kN/m2)'],3)
    new_df['Load(kN)'] =round(Cf*area*new_df['pz(kN/m2)']/storey_height,3)

    new_df = new_df.reset_index(drop=True)

    return new_df


def dynamic_wind_pressure(data_df: pd.DataFrame, height: float, category: str, Vb: float, k1: float, k3: float, k4: float,lx:float,
                         ly:float,Tx:float,Ty:float,beta:float,Cfx:float,Cfy:float,):
    """
    Returns a dataframe with dynamic wind pressure based on building heights and wind categories
    as per IS 875 Part-3:2015

    Parameters:
    - data_df : DataFrame containing the building storey height data.
    - height: Total height of the building.
    - category: Wind speed category.
    - Vb : Basic wind speed
    - k1, k3, k4: Environmental and structural factors affecting wind load.

    """

    # Validate required columns
    required_columns = ['Height']
    if not all(col in data_df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {', '.join([col for col in required_columns if col not in data_df.columns])}")

    # Initialize new columns with defaults or calculations
    data_df['Abs height'] = 0.0
    data_df['Avg height'] = 0.0
    data_df.loc[0, 'Abs height'] = height
    data_df.loc[0, 'Avg height'] = data_df.loc[0, 'Height'] / 2

    # Compute subsequent 'Abs height' and 'Avg height' values
    for i in range(1, len(data_df)):
        data_df.loc[i, 'Abs height'] = data_df.loc[i - 1, 'Abs height'] - data_df.loc[i-1, 'Height']
        data_df.loc[i, 'Avg height'] = (data_df.loc[i - 1, 'Height'] / 2) + (data_df.loc[i, 'Height'] / 2)

    # Load external data for wind speed factors
    try:
        hourlyk2_df = pd.read_csv("Hourly mean wind speed factor k2.csv")
    except FileNotFoundError:
        raise FileNotFoundError("The file 'Hourly mean wind speed factor k2.csv' was not found.")

    # Interpolate k2 values based on 'Abs height'
    data_df['k2i'] = np.round(np.interp(data_df['Abs height'], hourlyk2_df['Height'], hourlyk2_df[category]), 2)

    # Calculate wind pressures
    data_df['Vz,d(m/s)'] = np.round(Vb * k1 * data_df['k2i'] * k3 * k4, 2)
    data_df['pz(N/m2)'] = np.round(0.6 * data_df['Vz,d(m/s)']**2, 2)

    if (category=='Category 4'):
        Lh=70*(height/10)**0.25
    else:
        Lh=85*(height/10)**0.25
    if (category=='Category 1' or category=='Category 2'):
        gv=3.0
    else:
        gv=4.0
    Hs=1+((data_df['Abs height']/height)**2)
    Ihi=turbulence_intensity(category,z=data_df['Abs height'])
    r=2*Ihi  

    fx=1/Tx
    gRx=(2*math.log((3600*fx)))**0.5
    fy=1/Ty
    gRy=(2*math.log((3600*fy)))**0.5
    
    #X-dirn    
    
    Bsx=1/(1+(((0.26*(height-data_df['Abs height'])**2))+(0.46*(ly**2))**0.5)/Lh)
    Sx=1/((1+((3.5*fx*height)/data_df['Vz,d(m/s)']))*(1+((4*fx*ly)/data_df['Vz,d(m/s)'])))
    Nx=(fx*Lh)/data_df['Vz,d(m/s)']
    Ex=(math.pi*Nx)/((1+(70.8*Nx**2))**(5/6))
    phix=gv*Ihi*(Bsx**0.5)/2
    data_df['Gx']=round(1+r*((gv**2*Bsx*((1+phix)**2))+((Hs*gRx**2*Sx*Ex)/beta))**0.5,2)
    data_df['Storey shear,kN(Along X)']=round(Cfx*data_df['Gx']*data_df['Avg height']*ly*data_df['pz(N/m2)']*10**-3,2)
      

    #Y-dirn    
    
    Bsy=1/(1+(((0.26*(height-data_df['Abs height'])**2))+(0.46*(lx**2))**0.5)/Lh)
    Sy=1/((1+((3.5*fy*height)/data_df['Vz,d(m/s)']))*(1+((4*fy*lx)/data_df['Vz,d(m/s)'])))
    Ny=(fy*Lh)/data_df['Vz,d(m/s)']
    Ey=(math.pi*Ny)/((1+(70.8*Ny**2))**(5/6))
    phiy=gv*Ihi*(Bsy**0.5)/2
    data_df['Gy']=round(1+r*((gv**2*Bsy*((1+phiy)**2))+((Hs*gRy**2*Sy*Ey)/beta))**0.5,2)
    data_df['Storey shear,kN(Along y)']=round(Cfy*data_df['Gy']*data_df['Avg height']*lx*data_df['pz(N/m2)']*10**-3,2)
    
    # Drop any rows with NaN values that might have been introduced by improper data or missing values
    dyn_pressure_df = data_df.dropna()

    return dyn_pressure_df
