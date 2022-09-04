import pandas as pd
from DialoguesParser import DialoguesParser

if __name__ == '__main__':
    # load the data
    data = pd.read_csv(r'.\data\test_data.csv')

    # create DialoguesParser instance
    parser = DialoguesParser()

    # extract tags and concatenate them with the given data
    data_tags = parser.extract(data)
    tagged_data = pd.concat([data, data_tags], axis=1)

    # save the result
    tagged_data.to_csv('result.csv')

    # create report for each dialogue
    report = parser.get_reports_for_dialogues(data)

    # save the report
    report.to_csv('dialogues_report.csv')
