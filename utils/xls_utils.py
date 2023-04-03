"""
Description:
	Open and modify an exist excel file.

Author: lincoln12w
Github: https://github.com/Lincoln12w

Module:
  xlrd, xlwt, xlutils:
    Github: https://github.com/python-excel
    Doc: http://www.python-excel.org/ & http://xlwt.readthedocs.io/en/latest/
                                      & http://xlutils.readthedocs.io/en/latest/


"""

import xlrd
import xlwt
from xlutils.copy import copy

def xls_set_format():
    """
	Set cell format. Include font & alignment.
	"""
	# Set Font
    font = xlwt.Font()
    font.name = 'Times New Roman'
    font.bold = False
    font.underline = False
    font.italic = False

    # Set alignment
    alignment = xlwt.Alignment()

    alignment.horz = xlwt.Alignment.HORZ_CENTER
	# vert can be 'VERT_TOP', 'VERT_CENTER', 'VERT_BOTTOM', 'VERT_JUSTIFIED', 'VERT_DISTRIBUTED'
    alignment.vert = xlwt.Alignment.VERT_CENTER

    style = xlwt.XFStyle()
    style.font = font
    style.alignment = alignment

    return style

def xls_read_data(xls):
    """
    use 'xlrd' to read an exist excel.
    """
    # Load the excel file.
    workbook = xlrd.open_workbook(xls)

    # Locate the desired sheet.
    worksheet = workbook.sheets()[0]
    '''
    worksheets = workbook.sheet_names()							# get sheet names
    worksheet = workbook.sheet_by_name(u'Sheet1')				# get sheet by name
    worksheet = workbook.sheet_by_index(0)						# get sheet by index
    for worksheet_name in worksheets:							# iterate sheets
        worksheet = workbook.sheet_by_name(worksheet_name)
    '''

    # Read data from sheet.
    num_rows = worksheet.nrows
    for curr_row in range(num_rows):
        row = worksheet.row_values(curr_row)
        print('row'+curr_row+ 'is '+ row)  #print 'row%s is %s' % (curr_row, row) DK

    num_cols = worksheet.ncols
    for curr_col in range(num_cols):
        col = worksheet.col_values(curr_col)
        print('col'+curr_col+ 'is '+col) #print 'col%s is %s' % (curr_col, col) DK

    for rown in range(num_rows):
        for coln in range(num_cols):
            cell = worksheet.cell_value(rown, coln)
            print(cell) #print cell DK

def xls_write_data(xls, data):
    """
    use 'xlwt' to create a new excel.
    """
    # Create new book & new sheet
    workbook = xlwt.Workbook()
    sheet1 = workbook.add_sheet('sheet1', cell_overwrite_ok=True)
    sheet2 = workbook.add_sheet('sheet2', cell_overwrite_ok=True)

    sheet1.write(0, 0, 'write data')
    sheet1.write(0, 1, data)
    sheet2.write(0, 0, 'write data')
    sheet2.write(1, 2, data)

    workbook.save(xls)
    print('Done!')  #DK

def xls_append_data(xls, datas):
    """
    1. use 'xlrd' to read an exist excel; create a new xls file if not exist.
    2. use 'xlutils' to change the loaded workbook into 'xlwt' format workbook.
    3. use 'xlwt' to write the new copied workbook into filesystem.

    Params:
        xls: 	filename
        datas:	[['name1', value1], ['name2', value2], ...]
    """
    # Set up alignment.
    style = xls_set_format()
    cols = len(datas)

    # Open a xls file if existed or create a new one.
    try:
		# 'formatting_info=True' for kepping the cell format
        rd_workbook = xlrd.open_workbook(xls, formatting_info=True)
        wr_workbook = copy(rd_workbook)
        row = rd_workbook.sheets()[0].nrows
    except IOError:
        wr_workbook = xlwt.Workbook()
        sheet = wr_workbook.add_sheet('sheet', cell_overwrite_ok=True)

        for i in range(cols):
            sheet.write(0, i, datas[i][0], style)
        row = 1

    # Save datas into the first sheet of xls file.
    worksheet = wr_workbook.get_sheet(0)
    for i in range(cols):
        worksheet.write(row, i, str(datas[i][1]), style)

    wr_workbook.save(xls)

def append_analysis(index, name, res, types, filename, extradata=''):
    """
    Append analysis.
    """
    data = [['blank', '']] * index
    data.append(['data name', name])
    for i, t in enumerate(types):
        data.append([t, res[i]])
    data.append(['extra', extradata])
    xls_append_data(filename, data)

if __name__ == "__main__":
    xls_append_data('hello.xls', [['data', 100]])
