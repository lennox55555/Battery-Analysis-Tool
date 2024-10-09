# import necessary libraries for file operations, data analysis, visualization, and date manipulation
import os  # used for interacting with the operating system, such as file path management
import pandas as pd  # used for data manipulation and analysis with DataFrames
import numpy as np  # provides support for numerical operations and array handling
import matplotlib.pyplot as plt  # used for creating static plots and charts
import seaborn as sns  # used for creating attractive statistical visualizations
from datetime import timedelta  # used for time calculations, such as adding or subtracting time
import plotly.graph_objects as go  # used for creating interactive visualizations with Plotly
from plotly.subplots import make_subplots  # used for creating subplots with Plotly


def load_and_clean_battery_data(file_path):
    """
    Loads and cleans battery data from a CSV file.

    Params:
    file_path (str): Path to the CSV file containing battery data.

    Returns:
    pd.DataFrame: Cleaned DataFrame containing battery data with 'Time' as index.
    """
    # loads the raw data from CSV; skips first row and uses no predefined header to accommodate custom column names
    df = pd.read_csv(file_path, skiprows=1, header=None, low_memory=False)

    # sets column names using the first row of data; assigns meaningful labels to each column for easier reference
    df.columns = df.iloc[0]

    # removes the row used for column names; resets the index to start from 0 for consistency
    df = df.iloc[1:].reset_index(drop=True)

    # converts 'Time' column to datetime format; ensures proper handling of time-based operations in analysis
    df['Time'] = pd.to_datetime(df['Time'])

    # sets 'Time' as the index of the DataFrame; allows for time-based data access and manipulation
    df.set_index('Time', inplace=True)

    # identifies columns that should be numeric; finds columns currently stored as strings for conversion
    numeric_columns = df.columns[df.dtypes == 'object']

    # converts identified columns to numeric type; replaces non-numeric values with NaN to handle errors gracefully
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # returns the cleaned DataFrame for further analysis or visualization
    return df


def calculate_dynamic_thresholds(window_data, voltage_columns, deviation_factor=0.05):
    """
    Calculates dynamic thresholds for voltage data.

    Params:
    window_data (pd.DataFrame): DataFrame containing voltage data for a specific time window.
    voltage_columns (list): List of column names containing voltage data.
    deviation_factor (float): Factor to determine threshold range.

    Returns:
    pd.DataFrame: DataFrame with average, upper_threshold, and lower_threshold columns.
    """

    # nested function to calculate the average of top 5 values in a row, excluding the lowest to minimize outlier impact
    def avg_top_5(row):
        # sorts values in ascending order to exclude the lowest outliers for a more robust mean calculation
        sorted_values = np.sort(row.values)
        # calculates mean of the top 5 values to smooth data and reduce influence of extreme values
        return np.mean(sorted_values[1:6])

    # computes average voltage across specified columns using the top 5 method for each row to derive a smooth baseline
    avg_voltage = window_data[voltage_columns].apply(avg_top_5, axis=1)

    # calculates upper voltage limit based on the deviation factor; defines the upper bound of acceptable voltage range
    upper_threshold = avg_voltage * (1 + deviation_factor)

    # calculates lower voltage limit based on the deviation factor; defines the lower bound of acceptable voltage range
    lower_threshold = avg_voltage * (1 - deviation_factor)

    # returns a DataFrame containing the calculated average and dynamic thresholds for each time point in the window
    return pd.DataFrame({
        'average': avg_voltage,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold
    })


def calculate_rate_of_change(series, window='10min'):
    """
    Calculates the rate of change for a time series.

    Params:
    series (pd.Series): Time series data to calculate rate of change.
    window (str): Time window for rolling calculation.

    Returns:
    pd.Series: Rate of change in volts per hour.
    """
    # computes the change in voltage over a rolling window; smooths short-term fluctuations in the data
    # multiplies by 6 to convert the change from per minute to per hour for consistency in measurement units
    return series.diff().rolling(window=window).mean() * 6


def plot_min_voltages_with_dynamic_thresholds(df, file_name, window_hours=12, deviation_factor=0.05):
    """
    plots minimum cell voltages and identifies anomalies using dynamic thresholds.
    """
    min_voltage_columns = [col for col in df.columns if 'min_cell_voltage' in col]

    total_duration = df.index[-1] - df.index[0]
    num_windows = int(total_duration.total_seconds() / (window_hours * 3600)) + 1

    anomalies = []
    anomaly_plots = []

    for window in range(num_windows):
        start_time = df.index[0] + timedelta(hours=window * window_hours)
        end_time = start_time + timedelta(hours=window_hours)

        window_data = df.loc[start_time:end_time]

        if window_data.empty:
            continue

        thresholds = calculate_dynamic_thresholds(window_data, min_voltage_columns, deviation_factor)
        rates_of_change = pd.DataFrame({col: calculate_rate_of_change(window_data[col]) for col in min_voltage_columns})

        high_slope_mask = (rates_of_change.abs() > 0.0019).any(axis=1)

        expanded_high_slope_mask = pd.Series(False, index=window_data.index)
        for idx in high_slope_mask[high_slope_mask].index:
            expanded_high_slope_mask |= (window_data.index >= idx - timedelta(minutes=15)) & (
                        window_data.index <= idx + timedelta(minutes=15))

        low_voltage_mask = (window_data[min_voltage_columns] >= -0.25) & (window_data[min_voltage_columns] <= 0.25)
        any_low_voltage = low_voltage_mask.any(axis=1)
        service_downtime_mask = pd.Series(False, index=window_data.index)
        for idx in window_data[any_low_voltage].index:
            service_downtime_mask |= (window_data.index >= idx - timedelta(hours=2)) & (
                        window_data.index <= idx + timedelta(hours=2))

        window_anomalies = []

        fig = go.Figure()

        for column in min_voltage_columns:
            fig.add_trace(go.Scatter(x=window_data.index, y=window_data[column], mode='lines', name=column))

            anomaly_mask = (window_data[column] < thresholds['lower_threshold']) | (
                        window_data[column] > thresholds['upper_threshold'])
            true_anomaly_mask = anomaly_mask & ~service_downtime_mask & ~expanded_high_slope_mask

            anomaly_periods = window_data[true_anomaly_mask].groupby((~true_anomaly_mask).cumsum())
            for _, period in anomaly_periods:
                if len(period) > 0:
                    max_roc = rates_of_change.loc[period.index, column].abs().max()
                    if max_roc > 0.00005:
                        anomaly = {
                            'Rack': column,
                            'Start_Time': period.index[0],
                            'End_Time': period.index[-1],
                            'Duration': period.index[-1] - period.index[0],
                            'Max_RoC': max_roc,
                            'Type': 'Malfunction'
                        }
                        anomalies.append(anomaly)
                        window_anomalies.append(anomaly)
                        fig.add_trace(go.Scatter(
                            x=[period.index[0], period.index[-1]],
                            y=[period[column].iloc[0], period[column].iloc[-1]],
                            mode='markers',
                            marker=dict(color='red', size=10),
                            name='Malfunction',
                            showlegend=False
                        ))

        downtime_periods = window_data[service_downtime_mask].groupby((~service_downtime_mask).cumsum())
        for _, period in downtime_periods:
            if len(period) > 0:
                downtime = {
                    'Rack': 'All',
                    'Start_Time': period.index[0],
                    'End_Time': period.index[-1],
                    'Duration': period.index[-1] - period.index[0],
                    'Max_RoC': None,
                    'Type': 'Service Downtime'
                }
                anomalies.append(downtime)
                window_anomalies.append(downtime)
                fig.add_vrect(
                    x0=period.index[0], x1=period.index[-1],
                    fillcolor="blue", opacity=0.3,
                    layer="below", line_width=0,
                    name='Service Downtime'
                )

        fig.add_trace(
            go.Scatter(x=thresholds.index, y=thresholds['upper_threshold'], mode='lines', name='Upper Threshold',
                       line=dict(color='red', dash='dash')))
        fig.add_trace(
            go.Scatter(x=thresholds.index, y=thresholds['lower_threshold'], mode='lines', name='Lower Threshold',
                       line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=thresholds.index, y=thresholds['average'], mode='lines', name='Average of Top 5',
                                 line=dict(color='green', dash='dot')))

        fig.update_layout(
            title=f"{file_name}: Minimum Cell Voltages by Rack with Dynamic Thresholds ({start_time} to {end_time})",
            xaxis_title="Time",
            yaxis_title="Voltage (V)",
            legend_title="Rack",
            hovermode="x unified"
        )

        if window_anomalies:
            anomaly_plots.append(fig)

    return anomalies, anomaly_plots




def process_csv_files(input_folder, output_folder):
    """
    Processes all CSV files in the input folder and generates analysis results.

    Params:
    input_folder (str): Path to the folder containing CSV files.
    output_folder (str): Path to the folder where results will be saved.

    Returns:
    pd.DataFrame: Summary of detected anomalies and downtimes across all files.
    """
    # initializes lists to store anomalies and plots across all files for summary generation
    all_anomalies = []
    all_plots = []

    # lists all CSV files in the input folder for batch processing
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # processes each file one by one to extract and analyze battery data
    for i, csv_file in enumerate(csv_files, 1):
        # constructs the full path to the current file for loading
        file_path = os.path.join(input_folder, csv_file)
        # logs the progress of the file being processed for user feedback
        print(f"Processing file {i} of {len(csv_files)}: {csv_file}")

        try:
            # loads and cleans the data from the specified file for analysis
            df = load_and_clean_battery_data(file_path)

            # extracts the file name without extension for labeling plots
            file_name = os.path.splitext(csv_file)[0]

            # detects anomalies and generates plots using threshold analysis
            anomalies, plots = plot_min_voltages_with_dynamic_thresholds(df, file_name, deviation_factor=0.025)
            # aggregates anomalies and plots across all processed files
            all_anomalies.extend(anomalies)
            all_plots.extend(plots)

            # logs the number of anomalies and plots generated for the current file
            print(f"Finished processing {csv_file}. {len(anomalies)} anomalies detected. {len(plots)} plots generated.")
        # handles errors during file processing and logs them for troubleshooting
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

    # creates a DataFrame from all detected anomalies for summary and analysis
    anomaly_df = pd.DataFrame(all_anomalies)

    # checks if any anomalies were detected; if not, prepares an empty DataFrame for output
    if not anomaly_df.empty:
        # groups anomalies by 'Rack' and 'Type' for summary statistics
        summary = anomaly_df.groupby(['Rack', 'Type']).agg({
            'Start_Time': 'min',  # identifies the first occurrence of an anomaly
            'End_Time': 'max',  # identifies the last occurrence of an anomaly
            'Duration': 'sum',  # calculates total duration of anomalies
            'Max_RoC': lambda x: x.max() if x.notna().any() else None
            # finds the maximum rate of change for significant anomalies
        }).reset_index()

        # renames columns for clarity in the output summary
        summary.columns = ['Rack', 'Type', 'First anomaly time', 'Last anomaly time', 'Total duration',
                           'Max Rate of Change']

        # calculates the number of detected anomalies for each 'Rack' and 'Type' combination
        anomaly_counts = anomaly_df.groupby(['Rack', 'Type']).size().reset_index(name='Number of anomalies detected')
        # merges anomaly counts with the summary DataFrame for a complete view
        summary = summary.merge(anomaly_counts, on=['Rack', 'Type'])

        # reorders columns for better readability in the output summary
        summary = summary[['Rack', 'Type', 'Number of anomalies detected', 'First anomaly time', 'Last anomaly time',
                           'Total duration', 'Max Rate of Change']]

        # sorts the summary by 'Rack' and 'Type' for consistent output ordering
        summary = summary.sort_values(['Rack', 'Type'])
    else:
        # prepares an empty summary DataFrame with the required columns when no anomalies are detected
        summary = pd.DataFrame(
            columns=['Rack', 'Type', 'Number of anomalies detected', 'First anomaly time', 'Last anomaly time',
                     'Total duration', 'Max Rate of Change'])

    # saves the summary DataFrame as a CSV file for external review and documentation
    summary_csv_path = os.path.join(output_folder, 'combined_anomaly_summary.csv')
    summary.to_csv(summary_csv_path, index=False)
    print(f"\nCombined anomaly and downtime summary exported to: {summary_csv_path}")

    # checks if any plots were generated; if so, saves them as an interactive HTML file
    if all_plots:
        # defines the file path for saving all plots into a single HTML file
        html_path = os.path.join(output_folder, 'anomaly_plots.html')
        with open(html_path, 'w') as f:
            # writes basic HTML structure to contain Plotly graphs
            f.write('<html><head><title>Anomaly Plots</title></head><body>')
            # iterates through each Plotly figure and adds it to the HTML file
            for plot in all_plots:
                f.write(plot.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write('</body></html>')
        print(f"Interactive plots saved to: {html_path}")
        print(f"Total number of plots generated: {len(all_plots)}")
    else:
        # logs that no plots were generated if no anomalies were detected across all files
        print("No anomaly plots generated.")

    # returns the final summary DataFrame for potential use in other parts of the program
    return summary


# main execution block ensures that the following code only runs when this script is executed directly
if __name__ == "__main__":
    # sets paths for input folder containing CSV files and output folder for saving results
    input_folder = '/Users/lennox/Documents/Battery-Analysis-Tool'
    output_folder = '/Users/lennox/Documents/Battery-Analysis-Tool/output'

    # checks if the output folder exists; creates it if not to ensure results can be saved
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # logs the start of the analysis process to provide user feedback
    print("Starting analysis of all CSV files in the folder...")

    # processes all files and generates a summary of detected anomalies and downtimes
    anomaly_summary = process_csv_files(input_folder, output_folder)

    # displays the summary of detected anomalies and downtimes to the user for quick review
    print("\nAnalysis complete. Summary of detected anomalies and downtimes:")
    print(anomaly_summary)

    # informs the user that all plots have been saved and the analysis is complete
    print("\nAll plots have been generated and saved in the output folder.")
