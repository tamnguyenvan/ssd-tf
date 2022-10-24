#######################################################################################################################
# Logging framework for IXOlerator. Intended for use internally within Ixolerator.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_defs import *
from utils.ret_values import *
from connectors.interfaces.interface import *

import datetime

# Dump logs as configured (TBD)
class Log():
    def __init__(self, subsystem:str='meghnad', *args, **kwargs):
        if subsystem == 'meghnad':
            self.log_level = IXO_MEGHNAD_LOG_LEVEL
        elif subsystem == 'apps':
            self.log_level = IXO_APPS_LOG_LEVEL
        elif subsystem == 'connectors':
            self.log_level = IXO_CONNECTORS_LOG_LEVEL
        else:
            self.log_level = IXO_DEFAULT_LOG_LEVEL
        assert((self.log_level >= 0) and (self.log_level <= 4))

    def config_connectors(self, *args, **kwargs):
        self.connector_log = {}

    def CRITICAL(self, line_num:int, file_name:str, method_name:str, msg:str):
        if self.log_level >= IXO_LOG_CRITICAL:
            if IXO_LOG_EXPOSURE == 'internal':
                print("\n****CRITICAL**** Date: {},".format(str(datetime.datetime.now())),
                      "File: {},".format(str(file_name)),
                      "Method: {},".format(str(method_name)),
                      "Line: {}.\n".format(str(line_num)),
                      msg, "\n")
            else:
                print("\n****CRITICAL**** Date: {},".format(str(datetime.datetime.now())),
                      msg, "\n")

    def ERROR(self, line_num:int, file_name:str, method_name:str, msg:str):
        if self.log_level >= IXO_LOG_ERROR:
            if IXO_LOG_EXPOSURE == 'internal':
                print("\n***ERROR*** Date: {},".format(str(datetime.datetime.now())),
                      "File: {},".format(str(file_name)),
                      "Method: {},".format(str(method_name)),
                      "Line: {}.\n".format(str(line_num)),
                      msg, "\n")
            else:
                print("\n***ERROR*** Date: {},".format(str(datetime.datetime.now())),
                      msg, "\n")

    def WARNING(self, line_num:int, file_name:str, method_name:str, msg:str):
        if self.log_level >= IXO_LOG_WARNING:
            if IXO_LOG_EXPOSURE == 'internal':
                print("\n**WARNING** Date: {},".format(str(datetime.datetime.now())),
                      "File: {},".format(str(file_name)),
                      "Method: {},".format(str(method_name)),
                      "Line: {}.\n".format(str(line_num)),
                      msg, "\n")
            else:
                print("\n**WARNING** Date: {},".format(str(datetime.datetime.now())),
                      msg, "\n")

    def STATUS(self, line_num:int, file_name:str, method_name:str, msg:str):
        if self.log_level >= IXO_LOG_STATUS:
            if IXO_LOG_EXPOSURE == 'internal':
                print("\n*STATUS* Date: {},".format(str(datetime.datetime.now())),
                      "File: {},".format(str(file_name)),
                      "Method: {},".format(str(method_name)),
                      "Line: {}.\n".format(str(line_num)),
                      msg, "\n")
            else:
                print("\n*STATUS* Date: {},".format(str(datetime.datetime.now())),
                      msg, "\n")

    def VERBOSE(self, line_num:int, file_name:str, method_name:str, msg:str):
        if self.log_level >= IXO_LOG_VERBOSE:
            if IXO_LOG_EXPOSURE == 'internal':
                print("\nVERBOSE Date: {},".format(str(datetime.datetime.now())),
                      "File: {},".format(str(file_name)),
                      "Method: {},".format(str(method_name)),
                      "Line: {}.\n".format(str(line_num)),
                      msg, "\n")
            else:
                print("\nVERBOSE Date: {},".format(str(datetime.datetime.now())),
                      msg, "\n")

