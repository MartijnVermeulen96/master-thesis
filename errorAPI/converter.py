import os
import pandas as pd
import time
from sqlalchemy import create_engine
from datetime import datetime
import contextlib

class Converter:
    @staticmethod
    def convert(file, output, convert_to='csv'):
        if isinstance(file, pd.DataFrame):
            df = file

        if isinstance(file, str):
            if file.endswith('.tsv'):
                df = pd.from_csv(file, sep='\t')
            if file.endswith('.csv'):
                df = pd.from_csv(file)
            if file.endswith('.xlsx'):
                df = pd.from_excel(file)

        df.to_csv(output)