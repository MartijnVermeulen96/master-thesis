from errorAPI.tool import Tool

class FDchecker(Tool):
    def __init__(self, configuration):
        print("Creating FDchecker")
        super().__init__("FDchecker", configuration)

    def run(self, d):
        outputted_cells = {}
        for l_attribute, r_attribute in self.configuration:
            jl = d.dataframe.columns.get_loc(l_attribute)
            jr = d.dataframe.columns.get_loc(r_attribute)
            value_dictionary = {}
            for i, row in d.dataframe.iterrows():
                if row[l_attribute] not in value_dictionary:
                    value_dictionary[row[l_attribute]] = {}
                value_dictionary[row[l_attribute]][row[r_attribute]] = 1
            for i, row in d.dataframe.iterrows():
                if len(value_dictionary[row[l_attribute]]) > 1:
                    outputted_cells[(i, jl)] = ""
                    outputted_cells[(i, jr)] = ""

        return outputted_cells