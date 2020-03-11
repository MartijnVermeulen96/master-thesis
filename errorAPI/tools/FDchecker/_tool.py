from errorAPI.tool import Tool
from ...helpers import AutoFD
from ... import default_placeholder

class FDchecker(Tool):
    default_configuration = {"Auto": "FDTool"}
    def __init__(self, configuration):
        super().__init__("FDchecker", configuration)

    def run(self, d):
        outputted_cells = {}
        if "FDs" in self.configuration:
            fds = self.configuration["FDs"]
        else:
            fds = []

        if "Auto" in self.configuration:
            autofd_helper = AutoFD(self.configuration["Auto"])
            
            fds_auto = autofd_helper.run(d)
            fds.extend(fds_auto)
        
        for l_attribute, r_attribute in fds:
            # jl = d.dataframe.columns.get_loc(l_attribute)
            jr = d.dataframe.columns.get_loc(r_attribute)
            value_dictionary = {}
            if isinstance(l_attribute, str):
                l_attribute = (l_attribute)

            for i, row in d.dataframe.iterrows():
                row_val = tuple(row[col] for col in l_attribute)
                
                if row_val not in value_dictionary:
                    value_dictionary[row_val] = {}
                value_dictionary[row_val][row[r_attribute]] = 1

            for i, row in d.dataframe.iterrows():
                row_val = tuple(row[col] for col in l_attribute)
                if len(value_dictionary[row_val]) > 1:
                    # outputted_cells[(i, jl)] = ""
                    outputted_cells[(i, jr)] = default_placeholder

        return outputted_cells