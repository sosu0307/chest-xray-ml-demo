import pandas as pd


def combine_dataframes(covid_filepath = './src/models/data/COVID.metadata.csv',
                       lung_opacity_filepath = './src/models/data/Lung_Opacity.metadata.csv',
                       normal_filepath = './src/models/data/Normal.metadata.csv',
                       viral_pneumonia_filepath = './src/models/data/Viral_Pneumonia.metadata.csv'):
    """
    Combines all dataframes
    
    Args:
        covid_filepath: filepath to covid csv
        lung_opacity_filepath: filepath to lung opacity csv
        normal_filepath: filepath to normal csv
        viral_pneumonia_filepath: filepath to viral pneumonia csv

    Returns:
        covid_df = dataframe created from covid csv
        lung_opacity_df = dataframe created from lung opacity csv
        normal_df = dataframe created from normal csv
        viral_pneumonia_df = dataframe created from viral pneumonia csv
        combined_df = all dataframes concatenated
    """

    covid_df = pd.read_csv(covid_filepath)
    lung_opacity_df = pd.read_csv(lung_opacity_filepath)
    normal_df = pd.read_csv(normal_filepath)
    viral_pneumonia_df = pd.read_csv(viral_pneumonia_filepath)

    covid_df['diagnosis'] = 'covid'
    lung_opacity_df['diagnosis'] = 'lung_opacity'
    normal_df['diagnosis'] = 'normal'
    viral_pneumonia_df['diagnosis'] = 'viral_pneumonia'

    combined_df = pd.concat([covid_df, lung_opacity_df, normal_df, viral_pneumonia_df])
    combined_df = pd.get_dummies(combined_df, columns=['diagnosis'])
    diagnoses_columns = ['diagnosis_covid', 'diagnosis_lung_opacity', 'diagnosis_normal', 'diagnosis_viral_pneumonia'] 
    combined_df[diagnoses_columns] = combined_df[diagnoses_columns].astype(int)

    return covid_df, lung_opacity_df, normal_df, viral_pneumonia_df, combined_df
