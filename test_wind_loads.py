import wind_loads as wind_loads
import pandas as pd

def test_basic_wind_speed():

    location='Agra'
    structural_class='Temporary structures'
    vb,k1=wind_loads.basic_wind_speed(location,structural_class)
    assert vb,k1==(47, 0.71)

def test_terrain_factor():

    k2=wind_loads.terrain_factor(height=420,category='Category 3')
    assert k2==1.345

def test_area_averaging_factor():

    Ka=wind_loads.area_averaging_factor(22)
    assert Ka==0.95

def test_importance_factor():

    k4=wind_loads.importance_factor('Important Structures')
    assert k4==1.3

def test_wind_directionality_factor():

    Kd=wind_loads.wind_directionality_factor('Triangular')
    assert Kd==0.9

def test_design_wind_pressure():

    k3, Vz, pz, Kc, pd=wind_loads.design_wind_pressure(location='Agra',structural_class='Important structures', height=420,category='Category 3',slope=1.0,structure_imp='Important Structures',structural_form='Rectangular',tributary_area=22)
    assert (k3,Vz,pz,Kc,pd) == (1.0, 87.93, 4.64, 0.9, 3.25)

def test_load_calc_pressure():

    F=wind_loads.load_calc_pressure(Cpe=0.9,Cpi=0.1,area=20,pd=3.57)
    assert F==57.12

def test_load_calc_force():

    F=wind_loads.load_calc_force(Cf=1.2,area=20,pd=1.5)
    assert F==36.0

def test_damping_coefficient():

    beta=wind_loads.damping_coefficient('Welded steel structures')
    assert beta==0.01

def test_wind_data_frame():

    df=wind_loads.wind_data_frame(height=90,storey_height=3,category='Category 1',Vb=47,k1=1,k3=1,k4=1,
                    Kd=1,Ka=1,Kc=0.9,Cf=1.2,area=20)
    assert df[:3]['Height'].tolist() == [0, 3, 6]
    assert df[:3]['Vz(m/s)'].tolist()==[49.35,49.35,49.35]
    assert df[:3]['pd(kN/m2)'].tolist()==[1.315,1.315,1.315]
    assert df[:3]['Load(kN)'].tolist()==[11.688,11.688,11.688]

def test_turbulence_intensity():

    z_values=[pd.Series([2.5,2.5,3.0,3.0,3.0,3.0,3.0,])]
    expected_ti_values=[pd.Series([0.300,0.300,0.293,0.293,0.293,0.293,0.293])]

    for z, exp_ti in zip(z_values, expected_ti_values):
        ti = wind_loads.turbulence_intensity(category='Category 3', z=z)
        assert ti.equals(exp_ti)


def test_terrain_factor_hourly():

    hourlyk2=wind_loads.terrain_factor_hourly(height=25,category='Category 3')
    assert hourlyk2==0.615

def test_dynamic_wind_pressure():

    data={"Storey":['LMR','HR','Terrace','12F','11F','10F','9F'],"Height":[2.5,2.5,3,3,3,3,3]}
    data_df =pd.DataFrame(data)
    df=wind_loads.dynamic_wind_pressure(data_df,height=51.65,category='Category 3',Vb=55,k1=1,k3=1,k4=1.15,
                                        lx=10.4,ly=27.5,Tx=2.2,Ty=2.367,beta=0.02,Cfx=1.2,Cfy=1.2,)
    assert df[:3]['Abs height'].tolist() == [51.65,49.15,46.65]
    assert df[:3]['Avg height'].tolist()==[1.25,2.50,2.75]
    assert df[:3]['k2i'].tolist()==[0.70,0.70,0.69]
    assert df[:3]['Vz,d(m/s)'].tolist()==[44.28,44.28,43.64]
    assert df[:3]['pz(N/m2)'].tolist()==[1176.43,1176.43,1142.67]
    assert df[:3]['Gx'].tolist()==[3.41,3.41,3.37]
    assert df[:3]['Storey shear,kN(Along X)'].tolist()==[165.48,330.96,349.46]
    assert df[:3]['Gy'].tolist()==[3.79,3.78,3.73]
    assert df[:3]['Storey shear,kN(Along y)'].tolist()==[69.56,138.74,146.28]

