from typing import List

import pandas as pd
import re
from yargy.pipelines import morph_pipeline
from yargy.interpretation import fact
from yargy import Parser, rule, or_
from yargy.predicates import gram

GREETING_PATTERN = 'здравствуйте|(с )?добр(ый|ое|ым) (день|вечер|утром?)|привет(ствую)?'
GOODBYE_PATTERN = '(до (свидания|скорого))|(всего (доброго|хорошего))|((доброго|хорошего) (дня|вечера))|(пока)'
COMPANIES_SET = {'диджитал бизнес'}


class DialoguesParser:
    """
      A class used to represent a Parser

      Attributes
      ----------
      greeting_pattern : str
          regex pattern for greetings extraction
      goodbye_pattern : str
          regex pattern for goodbye extraction

      Methods
      -------
      get_tags_columns_names(self) -> List
          Getter for the private attribute __tags_columns_names.

      parse(dialogues: pd.DataFrame) -> pd.DataFrame
          Tags each manager line on containing greeting, name, company and goodbye.

      get_reports_for_dialogues(self, dialogues: pd.DataFrame,
                                  tagged: bool=False) -> pd.DataFrame
          Tags each dialogue on having parsed tags and on containing all obligatory
          parts (greeting and goodbye).
    """

    def __init__(self, greeting_pattern: str = GREETING_PATTERN,
                 goodbye_pattern: str = GOODBYE_PATTERN,
                 companies_set=COMPANIES_SET):
        """
        Attributes
        ----------
        greeting_pattern : str
            regex pattern for greetings extraction
        goodbye_pattern : str
            regex pattern for goodbye extraction
        companies_set: set
            set of possible companies
        """

        # yargi parsers
        self.name_parser = self.__create_name_parser()
        self.company_parser = self.__create_company_parser()

        # regex patterns
        self.greeting_pattern = re.compile(greeting_pattern)
        self.goodbye_pattern = re.compile(goodbye_pattern)
        self.companies_set_pattern = re.compile('(' + ')|('.join(companies_set) + ')')

        self.__tags_columns_names = ['greeting', 'manager_name', 'manager_self_represented',
                                     'company_name', 'goodbye']

    @property
    def get_tags_columns_names(self) -> List:
        """
        Get tags_columns_names
        """
        return self.__tags_columns_names

    @staticmethod
    def __create_name_parser():
        """
        Creates yargi parser for manager name extraction.
        Return:
          yargy.parser.Parser
        """
        possible_name_prefixes = ['меня зовут', 'это',
                                  'менеджер', 'вам звонит',
                                  'вас беспокоит', 'я',
                                  'говорит']

        full_name_pattern = fact('Name', ['prefix', 'name'])

        prefix = morph_pipeline(possible_name_prefixes)
        name = rule(gram('Name'))
        full_name_pattern = rule(prefix,
                                 name.interpretation(full_name_pattern.name)
                                 ).interpretation(full_name_pattern)

        return Parser(full_name_pattern)

    @staticmethod
    def __create_company_parser():
        """
        Creates yargi parser for company extraction.
        Return:
          yargy.parser.Parser
        """

        possible_prefixes = ['компания', 'из компании',
                             'фирма', 'из фирмы',
                             'офис', 'из офиса',
                             'подразделение', 'из подразделения',
                             'компании',  # e.g. отдел продаж компании
                             'из']

        full_name_pattern = fact('Company', ['prefix', 'name'])

        prefix = morph_pipeline(possible_prefixes)
        name = or_(gram('Name'),
                   gram('NOUN'),
                   gram('Orgn'))
        full_company_pattern = rule(prefix,
                                    name.interpretation(full_name_pattern.name)
                                    ).interpretation(full_name_pattern)

        # create parser
        return Parser(full_company_pattern)

    def __parse_greetings(self, manager_speech: pd.Series) -> pd.Series:
        """
        Parses greetings in the manager speech.

        Args:
          manager_speech: pd.Series that contains only manager lines
                          in a particular dialogue

        Return:
          pd.Series that contains in each row 1 if a greeting was extracted there,
          0 otherwise; the indexes match the indexes of the corresponding lines
          in manager_speech.
        """

        result = []
        for line in manager_speech:
            greeting = re.search(self.greeting_pattern, line)

            if greeting:
                result.append(1)
            else:
                result.append(0)

        result.extend([0] * (len(manager_speech) - len(result)))

        return pd.Series(result, index=manager_speech.index)

    def __parse_name(self, manager_speech: pd.Series) -> pd.Series:
        """
        Parses manager name in the manager speech.

        Args:
          manager_speech: pd.Series that contains only manager lines
                          in a particular dialogue

        Return:
          pd.Series that contains in each row manager name if it was extracted there,
          0 otherwise; the indexes match the indexes of the corresponding lines
          in manager_speech.
        """

        result = []
        for line in manager_speech:
            name = self.name_parser.find(line)
            if name:
                result.append(name.fact.name)
            else:
                result.append(0)

        result.extend([0] * (len(manager_speech) - len(result)))

        return pd.Series(result, index=manager_speech.index)

    def __look_up_in_dict(self, line: str) -> List:
        """
        Searches for known company names
        """
        return re.findall(self.companies_set_pattern, line)

    def __parse_company(self, manager_speech: pd.Series) -> pd.Series:
        """
        Parses company name in the manager speech.

        Args:
          manager_speech: pd.Series that contains only manager lines \n
                          in a particular dialogue

        Return:
          pd.Series that contains in each row a company name if it was extracted there, \n
          0 otherwise; the indexes match the indexes of the corresponding lines \n
          in manager_speech.
        """

        result = []
        for line in manager_speech:
            name = self.__look_up_in_dict(line)

            if name:
                result.append(name[0])
                continue

            name = self.company_parser.find(line)
            if name:
                result.append(name.fact.name)
            else:
                result.append(0)
        result.extend([0] * (len(manager_speech) - len(result)))

        return pd.Series(result, index=manager_speech.index)

    def __parse_goodbye(self, manager_speech: pd.Series) -> pd.Series:
        """
        Parses goodbye in the manager speech.

        Args:
          manager_speech: pd.Series that contains only manager lines
                          in a particular dialogue

        return: pd.Series that contains in each row 1 if a goodbye was extracted there, \n
                0 otherwise; the indexes match the indexes of the corresponding lines \n
                in manager_speech.
        """

        result = []
        for line in manager_speech:
            goodbye = re.search(self.goodbye_pattern, line)

            if goodbye:
                result.append(1)
            else:
                result.append(0)
        result.extend([0] * (len(manager_speech) - len(result)))

        return pd.Series(result, index=manager_speech.index)

    def extract(self, dialogues: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the presence of greeting, manager name, company name and goodbye in
        each line of a manager speech in a particular dialogue.

        Args:
          dialogues: pd.DataFrame that represents lines in a client-manager phone conversation. \n
                     It has to contain dlg_id (dialogue id), role ('manager' or 'client') \n
                     and text (a line).

        Return:
                pd.DataFrame that contains columns that represent the presence of
                one of the following tags: greeting, manager name, company name and \n
                goodbye in each line of the given dialogues dataframe. \n
                The shape is (number of lines in the given dataframe, 5); \n
                the indexes match the indexes of the corresponding lines  \n
                in the given dataframe.
        """

        tags = pd.DataFrame(index=dialogues.index)
        tags[self.__tags_columns_names] = 0

        managers_lines = dialogues[dialogues['role'] == 'manager'].groupby('dlg_id')
        for name, manager_speech in managers_lines:
            # little preprocessing
            text = manager_speech['text'].apply(str.lower)

            # tagging
            tags.loc[manager_speech.index, 'greeting'] = self.__parse_greetings(text)
            tags.loc[manager_speech.index, 'goodbye'] = self.__parse_goodbye(text)
            tags.loc[manager_speech.index, 'manager_name'] = self.__parse_name(text)
            tags.loc[manager_speech.index, 'company_name'] = self.__parse_company(text)

        tags['manager_self_represented'] = [1 if name != 0 else 0 for name in tags['manager_name']]
        return tags

    @staticmethod
    def __add_name_attribute(report: pd.DataFrame,
                             tagged_dialogues: pd.DataFrame,
                             col: str) -> pd.DataFrame:
        """
        Creates columns with names for report
        """
        get_names = tagged_dialogues.groupby('dlg_id')[col].apply(lambda x: x != 0)
        names = tagged_dialogues[['dlg_id', col]][get_names]
        names.drop_duplicates(inplace=True)
        report = report.merge(names, on='dlg_id', how='left')
        return report

    def get_reports_for_dialogues(self, dialogues: pd.DataFrame,
                                  tagged: bool = False) -> pd.DataFrame:
        """
        Extracts the presence of greeting, manager name, company name and goodbye in
        each line of a manager speech in a particular dialogue.

        Args:
          dialogues: pd.DataFrame that represents lines in a client-manager phone conversation.
                     It has to contain dlg_id (dialogue id), role ('manager' or 'client') and text (a line).\n
                     If tagged=True the columns from the extract() method must be included.
          tagged (default=False): True if the dialogues contains the columns from the extract() method,
                  otherwise the function will invoke extract().

        Return:
                pd.DataFrame contains columns which represent the number of lines where
                each tag (greeting, manager name, manager_self_represented, company name and \n
                goodbye) was presented for each dialogue in the given dataframe. It also provides a check
                (columns check_passed) on having both obligatory parts: greeting and goodbye.\n
                The shape is (number of dialogues in the given dataframe, 6).
        """

        if not tagged:
            tagged_dialogues = self.extract(dialogues)
            tagged_dialogues['dlg_id'] = dialogues['dlg_id']
        else:
            tagged_dialogues = dialogues[['dlg_id'] + self.__tags_columns_names]

        # create report
        report = tagged_dialogues.groupby('dlg_id')[self.__tags_columns_names].sum()

        # add manager name
        report = self.__add_name_attribute(report, tagged_dialogues, 'manager_name')

        # add company name
        report = self.__add_name_attribute(report, tagged_dialogues, 'company_name')

        # add check on requirements
        check_passed = report[['greeting', 'goodbye']].all(axis=1)
        report['check_passed'] = pd.Series(check_passed, dtype=int)

        return report
