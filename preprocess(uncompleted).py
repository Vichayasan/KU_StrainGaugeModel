import pandas as pd
import math
df = pd.read_csv("D:/Python/Data_MS_14_08_2025/5x5cm_Example1_MFD13082025_Non-Conductive_Press_probe/9phr_5x5cm_Thk6mm_1loop1_dia30mm_1R_1Pos_spd282_PH5s_LH60s_5-50pct/9phr_5x5cm_Thk6mm_1loop1_dia30mm_1R_1Pos_spd282_PH5s_LH60s_5-50pct.csv",index_col=0)
# print(df)
df1 = df.sort_index(ascending=True)
# print(df1)
df2 = df1.iloc[:,[0,1,2,3,4,5,6,23]]
# print(df2)
df3 = df2.copy()
Vv = []
Rr = []
Ff = []
Pp = []
for i in range(0,2571):
    Vi = ((df3.iloc[i,6] / (2**16 / 2 - 1)) * 4.096)
    Vv.append(Vi)
    Ri = (((((df3.iloc[i,6] / (2**16 / 2 - 1)) * 4.096) * 100000) / (3.3 - ((df3.iloc[i,6] / (2**16 / 2 - 1)) * 4.096)))) / 1000
    Rr.append(Ri)
    Fi = (df3.iloc[i,7]/1000) * 9.8
    Ff.append(Fi)
    Pi = (df3.iloc[i,7] / math.pi*(15**2)/1000000)*1000
    Pp.append(Pi)
df3["Voltage(V)"] = Vv
df3["Resistance(kΩ)"] = Rr
df3["Force(N)"] = Ff
df3["Pressure(kPa)"] = Pp
df3.to_excel("9phr_5x5cm_Thk6mm_1loop1_dia30mm_1R_1Pos_spd282_PH5s_LH60s_5-50pct.xlsx")
df_4 = df3.copy()
i = 0
i_row_r0 = []
for index, rule in df_4.iterrows():
    # print(z)
    z_axis = rule["Z"]
    stat_col = rule["Status"]
    # print(stat_col)
    if z_axis != -0.144 and stat_col == "Run" or z_axis == -0.125: # not dynamics
        i_row_r0.append(index)
    i += 1
# print(i_row_r0)
df_4.drop(i_row_r0, inplace=True)
df_5 = df_4.copy()
z_now = []
group_z = []
j = 0
for i in range(2108):
    z_before = df_5.iloc[i, 3]
    z_now.append(z_before)
    if z_before != z_now[i-1]:
        j+=1
    group_z.append(j)
    # print(j)
    # print(f"z_before={z_before}\tz_now={z_now[i-1]}")
group_z.append(j)
    
# print(len(group_z))
df_4["step"] = group_z
df_4.to_excel('preprocess_LP1.xlsx')
df_6 = df_4.copy()
step_0_v = {}
step_0_r = {}
vAVG_0 = 0
rAVG_0 = 0
#directory value for each column depend on step
for index, rule in df_6.iterrows():
    step = rule["step"]
    force = rule["Force(N)"]
    pres = rule["Pressure(kPa)"]
    vol = rule["Voltage(V)"]
    # print(vol)
    rr = rule["Resistance(kΩ)"]
    if step not in step_0_v:
        # print(step)
        step_0_v[step] = []
        # print(step_0_v)
    step_0_v[step].append(vol)
        # output {step1:(vol1, vol2, ...), step2:(...), ...}
        
print(step_0_v[1])
