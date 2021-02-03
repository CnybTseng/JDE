import xlwt
import warnings
import os.path as osp
from xlrd import open_workbook
from xlutils.copy import copy

def build_excel(filename, head, sheetname, override=True):
    '''Build a new .xls file.
    
    Param
    -----
    filename: Filename of .xls to be built.
    head    : The table head of .xls.
    '''
    if osp.isfile(filename):
        if override:
            warnings.warn('{} will be overrided'.format(filename))
        else:
            return
    wb = xlwt.Workbook()
    sh = wb.add_sheet(sheetname)
    for i, h in enumerate(head):
        sh.write(0, i, h)
    wb.save(filename)

def append_excel(filename, values):
    '''Append one or multiple rows to a exists .xls file.
    
    Param
    -----
    filename: A exists .xls filename.
    values  : List of one or multiple row values. E.G.,
        [[1, 2, 3], [4, 5, 6]].
    '''
    wb = open_workbook(filename, formatting_info=True)
    row = wb.sheets()[0].nrows
    excel = copy(wb)
    table = excel.get_sheet(0)
    for rv in values:
        for col, v in enumerate(rv):
            table.write(row, col, v)
        row += 1
    excel.save(filename)