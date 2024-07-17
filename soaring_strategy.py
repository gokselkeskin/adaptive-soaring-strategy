import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
matplotlib.use("Qt5Agg")
import math
import scipy.stats as st
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from peakdetect import peakdetect
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.optimize import curve_fit
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D



class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

full_track_control = False
do_plots = False
do_corrected_thermals = False
circles = False
do_last_points = False
daily_plots = False

interval1 = 7
interval2 = 18
###C.ciconia interval 7-18


bin_limit = 50

defined_threshold = 30





df_main = pd.read_csv(
    r'C:\Users\36702\PycharmProjects\glidepolar\prepared_data\Ciconia ciconia_smooth_and_velocity_calculations_day_added.csv')

df_main["date_in_unix_original"] = df_main["date_in_unix"]



individuals = df_main["individuals"].unique()
individuals = [individual for individual in individuals if str(individual) != 'nan']



updated_all_individuals = pd.DataFrame()

basic_thermal_climbrate = []
basic_thermal_climbrate_err = []
basic_mean_horizontal_speed = []
basic_mean_horizontal_speed_err = []


horizontal_velocity_samples = []
vertical_velocity_samples = []
radius_samples = []
K_samples = []

raw_radius_samples = []
raw_K_samples = []
raw_Vxy_samples = []
raw_Vz_samples = []
name_indiviuals = []

for individual in individuals:

    df_ind = df_main.loc[(df_main.individuals == individual)]


    updated_all_days = pd.DataFrame()

    days = df_main["only_date"].unique()
    days = [day for day in days if str(day) != 'nan']

    for day in days:
        df = df_ind.loc[(df_ind.only_date == day)]

        df["to_discard"] = (abs(df["date_in_unix"].diff()) > 1.01)  ## 3.01 for A.v

        df["to_discard"] = df["to_discard"].where(df["to_discard"]).bfill(limit=5).fillna(0).astype(bool)

        df[df["to_discard"]] = np.nan


        df["heading"] = np.arctan2(df["Vx_MA"], df["Vy_MA"]) / np.pi * 180
        df.loc[(df.heading < 0), "heading"] = df["heading"] + 360
        df["K"] = np.abs(df["K"])
        df["R"] = 1 / df["K"]

        df["heading_thermal"] = df["heading"]

        df["time_delta"] = df["date_in_unix"].diff()
        if len(df) < 100:
            continue


        deltat = float("%.2f" % df["time_delta"].describe()[1])

        if np.isnan(deltat):
            continue

        threshold_value = defined_threshold / deltat

        df["thermal_check"] = 0
        df.loc[(abs(df['R']) < 100) & (df["vertical_speed"] > 0) & (df["horizontal_speed"] > 3) , 'thermal_check'] = 1


        rolling_window = int((3/ deltat))
        df["thermal_check"] = df["thermal_check"].rolling(rolling_window, center=True, win_type='gaussian').mean(std=1)



        df.loc[abs(df['thermal_check']) == 0, 'heading_thermal'] = np.nan



        thermals = df.loc[df['thermal_check'] > 0]



        if df["Vxy_MA"].describe()[1] < 1 or len(df) == 0:
            continue


        df["Original_X(m)_MA"], df["Original_Y(m)_MA"], df["Original_Z(m)_MA"] = df["X(m)_MA"], df["Y(m)_MA"], df[
            "Z(m)_MA"]

        if full_track_control:
            points = np.array([df["X(m)_MA"], df["Y(m)_MA"], df["Z(m)_MA"]], dtype='object').T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segments, cmap='jet')
            vario = df["vertical_speed"].dropna()
            lc.set_array(np.array(vario))
            # lc.set_array(np.array(df["vertical_speed"]))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.add_collection3d(lc)
            ax.set_xlim(df["X(m)_MA"].min() - 400, df["X(m)_MA"].max() + 400)
            ax.set_ylim(df["X(m)_MA"].min() - 400, df["X(m)_MA"].max() + 400)
            ax.set_zlim(df["Z(m)_MA"].min(), df["Z(m)_MA"].max() + 100)
            ax.plot(df["X(m)_MA"], df["Y(m)_MA"], 0, color="grey", zorder=0)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f"{individual}{day}")
            line = ax.add_collection(lc)
            fig.colorbar(line)
            plt.show()


        counter_number = 0
        is_found = True
        extract_series = pd.DataFrame()


        datapoints = []

        new_dataframe = pd.DataFrame()

        df["Xw"] = ""
        df["Yw"] = ""
        df["WindX"] = ""
        df["WindY"] = ""

        peak_counter_x = []
        peak_counter_y = []
        number_of_daily_thermals = []

        last_thermal_timestamps = []

        for idx in range(0, len(df['heading_thermal'])):
            heading_value = df.iloc[idx, df.columns.get_loc('heading_thermal')]
            # heading_value = df['heading'][idx]
            if math.isnan(heading_value):
                # print(f"nan found: {idx}")
                if counter_number >= threshold_value:
                    print(f"thermal found, day:{day}, range: {((idx) - counter_number)}:{(idx - 1)}")
                    number_of_daily_thermals.append(day)
                    extract_series = df[((idx) - counter_number):(idx - 1)]
                    extract_series.reset_index(inplace=True)
                    extract_series.loc[extract_series.index[-1:]] = np.nan
                    start_heading = extract_series['heading_thermal'][0]
                    last_start_heading = 0

                    extract_series["date_in_unix"] = extract_series["date_in_unix"] - extract_series["date_in_unix"][0]

                    extract_series["X_visualization"] = extract_series["X(m)_MA"]
                    extract_series["Y_visualization"] = extract_series["Y(m)_MA"]

                    extract_series["X(m)_MA"] = extract_series["X(m)_MA"] - extract_series["X(m)_MA"][0]
                    extract_series["Y(m)_MA"] = extract_series["Y(m)_MA"] - extract_series["Y(m)_MA"][0]

                    peaks_x = peakdetect(extract_series["Vx_MA"], lookahead=int(3 / deltat))  # / deltat))
                    peaks_y = peakdetect(extract_series["Vy_MA"], lookahead=int(3 / deltat))  # / deltat))



                    higherPeaks_x = np.array(peaks_x[0])
                    lowerPeaks_x = np.array(peaks_x[1])
                    higherPeaks_y = np.array(peaks_y[0])
                    lowerPeaks_y = np.array(peaks_y[1])



                    if do_plots:
                        fig_first, ax_arrr = plt.subplots(2)
                        ax_arrr[0].plot(extract_series["date_in_unix"], extract_series["Vx_MA"])
                        ax_arrr[0].scatter((higherPeaks_x[:, 0] * deltat), higherPeaks_x[:, 1])
                        ax_arrr[0].scatter((lowerPeaks_x[:, 0] * deltat), lowerPeaks_x[:, 1])
                        ax_arrr[0].set_xlabel("time")
                        ax_arrr[0].set_ylabel("Vx")
                        ax_arrr[1].plot(extract_series["date_in_unix"], extract_series["Vy_MA"])
                        ax_arrr[1].scatter((higherPeaks_y[:, 0] * deltat), higherPeaks_y[:, 1])
                        ax_arrr[1].scatter((lowerPeaks_y[:, 0] * deltat), lowerPeaks_y[:, 1])
                        ax_arrr[1].set_xlabel("time")
                        ax_arrr[1].set_ylabel("Vy")
                        plt.show(block=True)

                    if len(lowerPeaks_x) > len(higherPeaks_x):
                        lowerPeaks_x = lowerPeaks_x[:-1, :]
                    if len(lowerPeaks_x) < len(higherPeaks_x):
                        higherPeaks_x = higherPeaks_x[1:, :]

                    if (len(higherPeaks_x) != len(lowerPeaks_x) or len(lowerPeaks_x) == 0):
                        counter_number = 0
                        is_found = False
                        extract_series = pd.DataFrame()
                        continue


                    differencePeaks_x1 = (higherPeaks_x + lowerPeaks_x) / 2
                    differencePeaks_x = differencePeaks_x1
                    for index, value in enumerate(differencePeaks_x):
                        if index <= len(differencePeaks_x) - 2:
                            if np.absolute(differencePeaks_x[:, 0][index] - differencePeaks_x[:, 0][index + 1]) < 10:
                                # print(differencePeaks_x[index])
                                row = np.where((differencePeaks_x == differencePeaks_x[index]).all(axis=1))
                                # print(row)
                                differencePeaks_x = np.delete(differencePeaks_x, row, axis=0)
                        else:
                            break

                    range_mid_x = np.arange(int(min(differencePeaks_x[:, 0]) + 1),
                                            int(max(differencePeaks_x[:, 0]) - 1))


                    if len(higherPeaks_y) > len(lowerPeaks_y):
                        higherPeaks_y = higherPeaks_y[:-1, :]
                    if len(higherPeaks_y) < len(lowerPeaks_y):
                        lowerPeaks_y = lowerPeaks_y[1:, :]

                    if (len(higherPeaks_y) != len(lowerPeaks_y) or len(lowerPeaks_y) == 0):
                        counter_number = 0
                        is_found = False
                        extract_series = pd.DataFrame()
                        continue


                    differencePeaks_y1 = (higherPeaks_y + lowerPeaks_y) / 2

                    differencePeaks_y = differencePeaks_y1
                    for index, value in enumerate(differencePeaks_y):
                        if index <= len(differencePeaks_y) - 2:
                            if np.absolute(differencePeaks_y[:, 0][index] - differencePeaks_y[:, 0][index + 1]) < 10:

                                row = np.where((differencePeaks_y == differencePeaks_y[index]).all(axis=1))

                                differencePeaks_y = np.delete(differencePeaks_y, row, axis=0)
                        else:
                            break
                    range_mid_y = np.arange(int(min(differencePeaks_y[:, 0]) + 1),
                                            int(max(differencePeaks_y[:, 0]) - 1))

                    if len(range_mid_y) == 0 or len(range_mid_x) == 0:

                        counter_number = 0
                        is_found = False
                        extract_series = pd.DataFrame()
                        continue


                    if range_mid_y[0] >= range_mid_x[0] and range_mid_x[-1] <= range_mid_y[-1]:
                        shared_mid = np.arange(int(min(differencePeaks_y[:, 0]) + 1),
                                               int(max(differencePeaks_x[:, 0]) - 1))
                        # print("1")
                    elif range_mid_y[0] >= range_mid_x[0] and range_mid_x[-1] >= range_mid_y[-1]:
                        shared_mid = np.arange(int(min(differencePeaks_y[:, 0]) + 1),
                                               int(max(differencePeaks_y[:, 0]) - 1))
                        # print("2")

                    elif range_mid_y[0] <= range_mid_x[0] and range_mid_x[-1] <= range_mid_y[-1]:
                        shared_mid = np.arange(int(min(differencePeaks_x[:, 0]) + 1),
                                               int(max(differencePeaks_x[:, 0]) - 1))
                        # print("3")

                    elif range_mid_y[0] <= range_mid_x[0] and range_mid_x[-1] >= range_mid_y[-1]:
                        shared_mid = np.arange(int(min(differencePeaks_x[:, 0]) + 1),
                                               int(max(differencePeaks_y[:, 0]) - 1))
                        # print("4")
                    else:
                        counter_number = 0
                        is_found = False
                        extract_series = pd.DataFrame()
                        continue

                    if len(shared_mid) == 0:
                        counter_number = 0
                        is_found = False
                        extract_series = pd.DataFrame()
                        continue

                    if len(differencePeaks_x) > 3 and len(differencePeaks_y) > 3:
                        f2_x = interp1d(differencePeaks_x[:, 0], differencePeaks_x[:, 1], kind='cubic')
                        f2_y = interp1d(differencePeaks_y[:, 0], differencePeaks_y[:, 1], kind='cubic')
                        # print("51")

                    else:
                        f2_x = interp1d(differencePeaks_x[:, 0], differencePeaks_x[:, 1], kind='linear')
                        f2_y = interp1d(differencePeaks_y[:, 0], differencePeaks_y[:, 1], kind='linear')
                        # print("61")

                    for i in extract_series["date_in_unix"].index:
                        if i <= shared_mid[0]:
                            extract_series["Xw"][i] = 0
                            extract_series["WindX"][i] = 0
                        else:
                            if i <= shared_mid[-1]:
                                extract_series["Xw"][i] = extract_series["Xw"][i - 1] + deltat * f2_x(i)
                                extract_series["WindX"][i] = f2_x(i)
                            else:
                                extract_series["Xw"][i] = extract_series["Xw"][i - 1]
                                extract_series["WindX"][i] = extract_series["WindX"][i - 1]
                        # for i in extract_series["time"].index:
                        if i <= shared_mid[0]:
                            extract_series["Yw"][i] = 0
                            extract_series["WindY"][i] = 0
                        else:
                            if i <= shared_mid[-1]:
                                extract_series["Yw"][i] = extract_series["Yw"][i - 1] + deltat * f2_y(i)
                                extract_series["WindY"][i] = f2_y(i)
                            else:
                                extract_series["Yw"][i] = extract_series["Yw"][i - 1]
                                extract_series["WindY"][i] = extract_series["WindY"][i - 1]

                    extract_series["x_update"] = extract_series["X(m)_MA"] - extract_series["Xw"]
                    extract_series["y_update"] = extract_series["Y(m)_MA"] - extract_series["Yw"]
                    extract_series["x_update"].apply(lambda x: float(x))
                    extract_series["y_update"].apply(lambda x: float(x))
                    extract_series["x_update"] = extract_series["x_update"][shared_mid[0]:shared_mid[-1]]
                    extract_series["y_update"] = extract_series["y_update"][shared_mid[0]:shared_mid[-1]]
                    extract_series["x_update_MA"] = extract_series["x_update"].rolling(5, center=True,
                                                                                       win_type='gaussian').mean(std=1)
                    extract_series["y_update_MA"] = extract_series["y_update"].rolling(5, center=True,
                                                                                       win_type='gaussian').mean(std=1)


                    if do_corrected_thermals:
                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        ax.set_xlim(10, 400)
                        ax.set_ylim(10, 400)
                        ax.set_zlim(200, max(extract_series["Z(m)_MA"]))
                        ax.plot(extract_series["X(m)_MA"], extract_series["Y(m)_MA"], extract_series["Z(m)_MA"])
                        # ax.plot(extract_series["x_update"], extract_series["y_update"], extract_series["ele"], label="without wind")
                        ax.plot(extract_series["x_update_MA"], extract_series["y_update_MA"], extract_series["Z(m)_MA"])
                        ax.set_xlabel("X(m)")
                        ax.set_ylabel("Y(m)")
                        fig.legend()
                        fig.tight_layout()
                        plt.show(block=True)
                        # fig.savefig("3D_thermal.svg")

                    extract_series['Vx_update'] = extract_series['x_update_MA'].diff() / extract_series[
                        "date_in_unix"].diff()
                    extract_series['Vy_update'] = extract_series['y_update_MA'].diff() / extract_series[
                        "date_in_unix"].diff()
                    extract_series['Vx_update'] = extract_series['Vx_update'].apply(lambda x: float(x))
                    extract_series['Vy_update'] = extract_series['Vy_update'].apply(lambda x: float(x))
                    extract_series["Vxy_update"] = np.sqrt(
                        ((extract_series['Vx_update'] ** 2) + (extract_series['Vy_update'] ** 2)))

                    extract_series['AX_update'] = extract_series['Vx_update'].diff() / extract_series[
                        "date_in_unix"].diff()
                    extract_series['AY_update'] = extract_series['Vy_update'].diff() / extract_series[
                        "date_in_unix"].diff()

                    extract_series['K_update'] = ((extract_series['Vx_update'] * extract_series['AY_update']) - (
                            extract_series['Vy_update'] * extract_series['AX_update'])) / (((extract_series[
                                                                                                 'Vx_update'] ** 2) + (
                                                                                                    extract_series[
                                                                                                        'Vy_update'] ** 2)) ** 1.5)
                    extract_series['K_update'] = np.abs(extract_series['K_update'])
                    extract_series['R_update'] = 1 / extract_series['K_update']


                    new_dataframe = pd.concat([new_dataframe, extract_series])

                    if do_plots:
                        fig11, ax_arrr = plt.subplots(3)
                        ax_arrr[0].plot(extract_series["date_in_unix"], extract_series["Vx_update"], "--o")
                        ax_arrr[0].set_ylabel("Vx")
                        ax_arrr[0].set_xlabel("time")
                        ax_arrr[1].plot(extract_series["date_in_unix"], extract_series["Vy_update"], "--o", )
                        ax_arrr[1].set_ylabel("Vy")
                        ax_arrr[1].set_xlabel("time")
                        ax_arrr[2].plot(extract_series["date_in_unix"], extract_series["R_update"], "--o")
                        ax_arrr[2].set_ylabel("R")
                        ax_arrr[2].set_xlabel("time")
                        fig11.tight_layout()
                        fig11.legend()
                        plt.show(block=True)

                        fig1, ax_arrr = plt.subplots(2, 2)
                        ax_arrr[0, 0].plot(extract_series["date_in_unix"], extract_series["Vx_MA"], "--o")
                        ax_arrr[0, 0].plot((higherPeaks_x[:, 0] * deltat), higherPeaks_x[:, 1], c="g", marker="o",
                                           label="Vx Local Extreme")
                        ax_arrr[0, 0].plot((lowerPeaks_x[:, 0] * deltat), lowerPeaks_x[:, 1], c="g", marker="o")
                        ax_arrr[0, 0].plot((differencePeaks_x[:, 0] * deltat), differencePeaks_x[:, 1], c="r",
                                           marker="o",
                                           label="Wind Average")
                        ax_arrr[0, 0].plot(range_mid_x * deltat, f2_x(range_mid_x), c="k", label="Wind Estimation")

                        ax_arrr[0, 0].legend()
                        ax_arrr[0, 0].set_xlabel("time(s)")
                        ax_arrr[0, 0].set_ylabel("Vx (m/s)")

                        ax_arrr[0, 1].plot(extract_series["date_in_unix"], extract_series["Xw"], "--o")

                        ax_arrr[1, 0].plot(extract_series["date_in_unix"], extract_series["Vy_MA"], "--o")
                        ax_arrr[1, 0].plot((higherPeaks_y[:, 0] * deltat), higherPeaks_y[:, 1], c="g", marker="o",
                                           label="Vy Local Extreme")

                        ax_arrr[1, 0].plot((lowerPeaks_y[:, 0] * deltat), lowerPeaks_y[:, 1], c="g", marker="o")
                        ax_arrr[1, 0].plot((differencePeaks_y[:, 0] * deltat), differencePeaks_y[:, 1], c="r",
                                           marker="o",
                                           label="Wind Average")
                        ax_arrr[1, 0].plot(range_mid_y * deltat, f2_y(range_mid_y), c="k", label="Wind Estimation")
                        ax_arrr[1, 0].legend()
                        ax_arrr[1, 0].set_xlabel("time(s)")
                        ax_arrr[1, 0].set_ylabel("Vy (m/s)")

                        ax_arrr[1, 1].plot(extract_series["date_in_unix"], extract_series["Yw"], "--o")

                        ax_arrr[0, 1].set_title("drift on X axis")
                        ax_arrr[0, 1].set_xlabel("time(s)")
                        ax_arrr[0, 1].set_ylabel("drift(m)")

                        ax_arrr[1, 1].set_title("drift on Y axis")
                        ax_arrr[1, 1].set_xlabel("time(s)")
                        ax_arrr[1, 1].set_ylabel("drift(m)")

                        fig1.tight_layout()
                        plt.show(block=True)
                        # fig1.savefig("wind.svg")

                    thermal_last_timestamps = extract_series["date_in_unix_original"].iloc[-2]
                    last_thermal_timestamps.append(thermal_last_timestamps)

                    extract_series = extract_series.dropna()
                    extract_series.reset_index(inplace=True)

                    # last_thermal_timestamp = extract_series["date_in_unix"].iloc[-1]

                    for series_idx in range(1, len(extract_series['heading'])):
                        if math.fabs(start_heading - extract_series['heading'][series_idx]) <= 35.0 and (
                                series_idx - last_start_heading >= 10 / deltat):
                            pattern_series = extract_series[last_start_heading: series_idx]

                            if circles:
                                fig_circles = plt.figure()
                                plt.plot(pattern_series["x_update"], pattern_series["y_update"])
                                plt.axis('equal')
                                plt.show(block=True)

                            horizontal_velocity_mean = np.mean(pattern_series["Vxy_update"])
                            vertical_velocity_mean = np.mean(pattern_series["Vz"])
                            radius_mean = np.mean(pattern_series["R_update"])
                            K_mean = np.mean(pattern_series["K_update"])

                            raw_radius_samples.append(pattern_series["R_update"].values)
                            raw_K_samples.append(pattern_series["K_update"].values)
                            raw_Vxy_samples.append(pattern_series["Vxy_update"].values)
                            raw_Vz_samples.append(pattern_series["Vz"].values)
                            name_indiviuals.append(pattern_series["individuals"].unique())

                            horizontal_velocity_samples.append(horizontal_velocity_mean)
                            vertical_velocity_samples.append(vertical_velocity_mean)
                            radius_samples.append(radius_mean)
                            K_samples.append(K_mean)
                            datapoints.append(len(pattern_series))

                            # raw_radius_samples.append(pattern_series["R_update"][series_idx])
                            # raw_K_samples.append(pattern_series["K_update"][series_idx])

                            last_start_heading = series_idx + 1
                            start_heading = extract_series['heading'][series_idx]

                # print("Reset!")
                counter_number = 0
                is_found = False
                extract_series = pd.DataFrame()
            else:
                counter_number += 1
                is_found = True

        cols = list(df.columns.values[0:-4])
        if len(new_dataframe) < 10:
            continue
        updated_df = df.join(new_dataframe.set_index(cols), on=cols, how='left', lsuffix='_left', rsuffix='_right')

        updated_df.loc[updated_df["WindX_right"] == 0, 'WindX_right'] = np.nan
        updated_df.loc[updated_df["WindY_right"] == 0, 'WindY_right'] = np.nan

        average_wind_x = np.mean(updated_df["WindX_right"])
        average_wind_y = np.mean(updated_df["WindY_right"])

        updated_df["air_speed_x"] = updated_df["Vx_MA"] - average_wind_x
        updated_df["air_speed_y"] = updated_df["Vy_MA"] - average_wind_y

        updated_df["air_speed_xy"] = np.sqrt(
            ((updated_df['air_speed_x'] ** 2) + (updated_df['air_speed_y'] ** 2)))

        thermal = updated_df[updated_df["Xw_right"].notnull()]
        thermal = thermal[thermal["vertical_speed"] > 0]
        basic_thermal_climbrate.append(np.mean(thermal["vertical_speed"]))
        basic_thermal_climbrate_err.append(st.sem(thermal["vertical_speed"], ddof=1))

        gliding_part = updated_df[updated_df["vertical_speed"] < 0]
        gliding_part = gliding_part[gliding_part["K"] < 0.01]






        basic_mean_horizontal_speed.append(np.mean(gliding_part["air_speed_xy"]))
        basic_mean_horizontal_speed_err.append(st.sem(gliding_part["air_speed_xy"], ddof=1))


        if daily_plots:
            fig_work_parts = plt.figure()
            ax = fig_work_parts.gca(projection='3d')

            gliding_part["to_discard"] = (abs(gliding_part["date_in_unix"].diff()) > 1.01)
            gliding_part[gliding_part["to_discard"]] = np.nan

            thermal["to_discard"] = (abs(thermal["date_in_unix"].diff()) > 1.01)
            thermal[thermal["to_discard"]] = np.nan



            ax.plot(df["Original_X(m)_MA"], df["Original_Y(m)_MA"], df["Original_Z(m)_MA"], color="Green")

            ax.plot(gliding_part["X(m)_MA"], gliding_part["Y(m)_MA"], gliding_part["Z(m)_MA"], color="Blue",
                    label="Gliding")
            ax.plot(thermal["X_visualization"], thermal["Y_visualization"], thermal["Z(m)_MA"], color="Red",
                    label="Thermals")



            first_valid_index = df["Original_X(m)_MA"].first_valid_index()

            x_metric_difference = gliding_part["X(m)_MA"].max() - gliding_part["X(m)_MA"].min()
            y_metric_difference = gliding_part["Y(m)_MA"].max() - gliding_part["Y(m)_MA"].min()

            x_metric_difference = x_metric_difference / 10
            y_metric_difference = y_metric_difference / 10

            ax.arrow3D(int(df["Original_X(m)_MA"][first_valid_index]), int(df["Original_Y(m)_MA"][first_valid_index]),
                       int(df["Original_Z(m)_MA"].max()) + 100, average_wind_x * x_metric_difference,
                       average_wind_y * y_metric_difference, 0,
                       mutation_scale=20,
                       arrowstyle="-|>", color="black", zorder=30, label="Wind")
            #

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')

            ax.set_title(f"{individual}{day}")

            ax.legend()



            plt.tight_layout()
            plt.show(block=True)



        ### daily csv##########################################
        daily_record = False
        if daily_record:
            daily_csv_results = updated_df["Species"].unique()[0], updated_df["individuals"].unique()[0], \
                                updated_df["only_date"].unique()[0], len(
                number_of_daily_thermals), average_wind_x, average_wind_y, len(thermal),np.mean(thermal["vertical_speed"]), st.sem(
                thermal["vertical_speed"], ddof=1),len(gliding_part), np.mean(gliding_part["air_speed_xy"]), st.sem(
                gliding_part["air_speed_xy"], ddof=1)
            with open("daily_results_30secs.csv", "a+", newline="") as daily_result_updater:
                csv_writer = csv.writer(daily_result_updater)
                csv_writer.writerow(daily_csv_results)
                daily_result_updater.close()

        updated_all_days = pd.concat([updated_all_days, updated_df])
        if do_last_points:
            fig_circles = plt.figure()
            plt.scatter(updated_df["air_speed_xy"], updated_df["vertical_speed"])
            plt.scatter(updated_df["Vxy_MA"], updated_df["vertical_speed"])
            plt.title(f"{updated_df['Species'].unique()[0]}")
            # plt.axis('equal')
            plt.show(block=True)
    updated_all_individuals = pd.concat([updated_all_individuals, updated_all_days])

###### gliding filter##############


gliding = updated_all_individuals.loc[(updated_all_individuals.vertical_speed < 0)]
gliding = gliding.loc[(gliding.K < 0.01)]

meanYValues = []
stdYValues = []
semYValues = []
countYValues = []
meanXValues = []
medianYValues = []
modeYValues = []
raw_x = []
raw_y = []

for increment in np.arange(interval1, interval2, 1):
    logicalIndexs = np.logical_and((gliding["air_speed_xy"] >= increment), gliding["air_speed_xy"] < (increment + 1))
    foundIndexs = np.where(logicalIndexs)
    raw_x.append(gliding["air_speed_xy"])
    raw_y.append(gliding["vertical_speed"])
    valuesY = np.extract(logicalIndexs, gliding["vertical_speed"])
    valuesX = np.extract(logicalIndexs, gliding["air_speed_xy"])

    if len(valuesY) >= bin_limit:
        meanYValues.append(np.mean(valuesY))
        medianYValues.append(np.median(valuesY))
        modeYValues.append(st.mode(valuesY)[0])
        stdYValues.append(np.std(valuesY, ddof=1))
        countYValues.append(len(valuesY))
        semYValues.append(st.sem(valuesY, ddof=1))

        meanXValues.append(increment)


x = np.array(meanXValues)
y = np.array(meanYValues)
y_err = np.array(semYValues)


def f(x, a, b, c):
    return (a * (x ** 2)) + (b * x) + c


def f_linear(x, a, b):
    return (a * x) + b



import scipy as sc


constantsQuadErr, pcov = curve_fit(f, x, y)
sigma_fit = np.sqrt(np.diagonal(pcov))



a_mean = constantsQuadErr[0]
b_mean = constantsQuadErr[1]
c_mean = constantsQuadErr[2]


x_line = np.arange(5, 25, 1)

y_line_mean = f(x_line, a_mean, b_mean, c_mean)

B1 = 1
B2 = 2
B3 = 3
######################################## mean tangent ###################################

tangent_x1_0 = np.sqrt((c_mean - 0) / a_mean)
tangent_y1_0 = (2 * c_mean) + b_mean * tangent_x1_0

tangent_x1_1 = np.sqrt((c_mean - B1) / a_mean)
tangent_y1_1 = (2 * a_mean * tangent_x1_1 + b_mean) * tangent_x1_1 + B1

tangent_x1_2 = np.sqrt((c_mean - B2) / a_mean)
tangent_y1_2 = (2 * a_mean * tangent_x1_2 + b_mean) * tangent_x1_2 + B2

tangent_x1_3 = np.sqrt((c_mean - B3) / a_mean)
tangent_y1_3 = (2 * a_mean * tangent_x1_3 + b_mean) * tangent_x1_3 + B3

optimal_thermal_x = [tangent_x1_0, tangent_x1_1, tangent_x1_2, tangent_x1_3]
optimal_thermal_y = [0, 1, 2, 3]


MacCreadyConstant, _ = sc.optimize.curve_fit(f_linear, optimal_thermal_y, optimal_thermal_x)
mac_cready_fit_a, mac_cready_fit_b = MacCreadyConstant

optimal_y_line = np.arange(0, 3.1, 0.1)
optimal_x_line = f_linear(optimal_y_line, mac_cready_fit_a, mac_cready_fit_b)



basic_mean_horizontal_speed = np.array(basic_mean_horizontal_speed)
basic_thermal_climbrate = np.array(basic_thermal_climbrate)
basic_thermal_climbrate_err = np.array(basic_thermal_climbrate_err)
basic_mean_horizontal_speed_err = np.array(basic_mean_horizontal_speed_err)

where_are_nans = np.isnan(basic_mean_horizontal_speed_err)

basic_mean_horizontal_speed = basic_mean_horizontal_speed[~where_are_nans]
basic_thermal_climbrate = basic_thermal_climbrate[~where_are_nans]
basic_mean_horizontal_speed_err = basic_mean_horizontal_speed_err[~where_are_nans]
basic_thermal_climbrate_err = basic_thermal_climbrate_err[~where_are_nans]

where_are_nans2 = np.isnan(basic_thermal_climbrate_err)

basic_mean_horizontal_speed = basic_mean_horizontal_speed[~where_are_nans2]
basic_thermal_climbrate = basic_thermal_climbrate[~where_are_nans2]
basic_mean_horizontal_speed_err = basic_mean_horizontal_speed_err[~where_are_nans2]
basic_thermal_climbrate_err = basic_thermal_climbrate_err[~where_are_nans2]

df_daily = pd.DataFrame()

df_daily["daily_climbrate"] = basic_thermal_climbrate
df_daily["climbrate_error"] = basic_thermal_climbrate_err
df_daily["mean_of_gliding"] = basic_mean_horizontal_speed
df_daily["error_of_gliding"] = basic_mean_horizontal_speed_err

std_dev_of_horizontal_speed = np.std(df_daily["mean_of_gliding"].values)
mean_of_horizontal_speed = np.mean(df_daily["mean_of_gliding"].values)
df_daily["sigma"] = (df_daily["mean_of_gliding"].values - mean_of_horizontal_speed) / std_dev_of_horizontal_speed


df_daily = df_daily.loc[(df_daily.sigma <= 2)]
df_daily = df_daily.loc[(df_daily.sigma >= -2)]

df_daily = df_daily.reset_index()

average_climbrate = df_daily["daily_climbrate"].values
average_climbrate_err= df_daily["climbrate_error"].values
average_horizontal_speed = df_daily["mean_of_gliding"].values
average_horizontal_speed_err = df_daily["error_of_gliding"].values



y_linear_fit = np.arange(0, 3.1, 0.1)
linear_fit, _ = sc.optimize.curve_fit(f_linear, average_climbrate, average_horizontal_speed)
a_linear, b_linear = linear_fit
x_linear_fit = f_linear(y_linear_fit, a_linear, b_linear)


fig_last, axs = plt.subplots(nrows=1, ncols=1)

axs.errorbar(x, y, yerr=y_err, fmt="o", c="red")
axs.plot(x_line, y_line_mean, c="red", label="fit_on_mean")


axs.errorbar(basic_mean_horizontal_speed, basic_thermal_climbrate, xerr=basic_mean_horizontal_speed_err,
             yerr=basic_thermal_climbrate_err, fmt="o", c="orange", alpha=0.7)
axs.plot(x_linear_fit, y_linear_fit, c="orange")
axs.errorbar(average_horizontal_speed, average_climbrate, yerr= average_climbrate_err,xerr=average_horizontal_speed_err,fmt="o",color="orange")



axs.plot(optimal_x_line, optimal_y_line, c="red", alpha=0.75)
axs.plot([0, tangent_x1_0], [0, 0], ":", c="red", alpha=0.75)  # ,label="O dashed")
axs.plot([tangent_x1_0, tangent_x1_0], [tangent_y1_0, 0], ":", c="red", alpha=0.5)  # ,label="O dashed")
axs.plot([0, tangent_x1_1], [1, 1], ":", c="red", alpha=0.5)  # ,label="1 dashed")
axs.plot([tangent_x1_1, tangent_x1_1], [tangent_y1_1, 1], ":", c="red", alpha=0.5)  # ,label="1 dashed")
axs.plot([0, tangent_x1_2], [2, 2], ":", c="red", alpha=0.5)  # ,label="1 dashed")
axs.plot([tangent_x1_2, tangent_x1_2], [tangent_y1_2, 2], ":", c="red", alpha=0.5)  # ,label="1 dashed")
axs.plot([0, tangent_x1_3], [3, 3], ":", c="red", alpha=0.5)  # ,label="1 dashed")
axs.plot([tangent_x1_3, tangent_x1_3], [tangent_y1_3, 3], ":", c="red", alpha=0.5)
axs.plot([0, tangent_x1_0], [0, tangent_y1_0], "--", c="grey")  # ,label="O climbrate")
axs.plot([0, tangent_x1_1], [1, tangent_y1_1], "--", c="grey")  # ,label="1 climbrate")
axs.plot([0, tangent_x1_2], [2, tangent_y1_2], "--", c="grey")  # ,label="2 climbrate")
axs.plot([0, tangent_x1_3], [3, tangent_y1_3], "--", c="grey")
#

axs.grid()
axs.set_xlim([0, 20.5])
axs.set_ylim([-3, 3.5])
axs.set_title(f'Cross-County Optimisation Strategy of {df_main["Species"].unique()[0]}')
axs.set_ylabel('Vertical Speed (m/s)')
axs.set_xlabel('Horizontal Speed (m/s)')

fig_last.set_size_inches(5, 8)
plt.show(block=True)


########################### mean flight performance#########################################################################
x_best_glide_mean = np.sqrt(c_mean / a_mean)
y_best_glide_mean = (2 * c_mean) + b_mean * np.sqrt(c_mean / a_mean)
best_glide_ratio_mean = np.absolute(x_best_glide_mean / y_best_glide_mean)

# min sink
x_min_sink_mean = -b_mean / (2 * a_mean)
y_min_sink_mean = f(x_min_sink_mean, *constantsQuadErr)

results = df_main["Species"].unique()[0], a_mean, b_mean, c_mean, pcov, x_best_glide_mean, y_best_glide_mean, best_glide_ratio_mean, x_min_sink_mean, y_min_sink_mean, \
          a_linear, b_linear

record = False

if record:
    with open("Result_soaring_strategy.csv", "a+", newline="") as result_updater:
        csv_writer = csv.writer(result_updater)
        csv_writer.writerow(results)
        result_updater.close()

raw_radius_samples = np.hstack(raw_radius_samples)
raw_radius_samples = [ele for ele in raw_radius_samples if ele > 0]

raw_K_samples = np.hstack(raw_K_samples)
raw_Vxy_samples = np.hstack(raw_Vxy_samples)
raw_Vz_samples = np.hstack(raw_Vz_samples)
name_indiviuals = np.hstack(name_indiviuals)

mean_radius = np.mean(raw_radius_samples)
median_radius = np.median(raw_radius_samples)
mean_Vxy = np.mean(raw_Vxy_samples)
median_Vxy = np.median(raw_Vxy_samples)
mean_Vz = np.mean(raw_Vz_samples)
median_Vz = np.median(raw_Vz_samples)


