from errorAPI.tool import Tool
import re

class Regex(Tool):
    def __init__(self, configuration):
        print("Creating Regex")
        super().__init__("Regex", configuration)

    def run(self, d):
        print("I know a lot of patterns!")
        outputted_cells = {}
        for attribute, pattern, match_type in self.configuration:
            j = d.dataframe.columns.get_loc(attribute)
            for i, value in d.dataframe[attribute].iteritems():
                if match_type == "OM":
                    if len(re.findall(pattern, value, re.UNICODE)) > 0:
                        outputted_cells[(i, j)] = ""
                else:
                    if len(re.findall(pattern, value, re.UNICODE)) == 0:
                        outputted_cells[(i, j)] = ""

        return outputted_cells