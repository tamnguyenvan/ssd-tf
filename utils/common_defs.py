#######################################################################################################################
# Common definitions for IXOlerator. Intended for use internally within Ixolerator.
#
# Copyright: All rights reserved. Inxite Out. 2022.
#
# Author: Kaushik Bar
#######################################################################################################################

from utils.common_seed import *

# Log level values
IXO_LOG_CRITICAL = 0
IXO_LOG_ERROR = 1
IXO_LOG_WARNING = 2
IXO_LOG_STATUS = 3
IXO_LOG_VERBOSE = 4

# Log levels for subsystems
IXO_DEFAULT_LOG_LEVEL = IXO_LOG_WARNING
IXO_MEGHNAD_LOG_LEVEL = IXO_LOG_ERROR
IXO_APPS_LOG_LEVEL = IXO_LOG_STATUS
IXO_CONNECTORS_LOG_LEVEL = IXO_LOG_CRITICAL

IXO_LOG_EXPOSURE = 'internal'

# Decorator to generate headers for methods
def method_header(description:str,
                  arguments:str='None', returns:str='None', created_by:str='None',
                  examples:str='None', assumptions:str='None', others:str='None') -> object:
    def _add_doc_wrapper(f):
        f.__doc__ = f"""
Name: {f.__name__}
----  

Description:
-----------
{description}

Arguments:
---------        
{arguments}

Returns:
-------
{returns}

Created by:
-----------        
{created_by}

Examples:
--------        
{examples}

Assumptions:
-----------        
{assumptions}

Others:
-------------        
{others}"""
        return f

    return _add_doc_wrapper

# Decorator to generate headers for classes
def class_header(description:str) -> object:
    def _add_doc_wrapper(f):
        f.__doc__ = f"""{description}"""
        return f

    return _add_doc_wrapper

