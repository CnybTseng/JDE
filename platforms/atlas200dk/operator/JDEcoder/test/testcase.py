import os
import data_provider as dp

def example():
    # Create a new DataProvider.
    dp1 = dp.DataProvider()

    # You can add input data (bytes) to data provider.
    example_input_data = bytes([0x0a, 0x37, 0x13, 0x00])
    dp1.add_input_by_data(example_input_data)

    # Or you can add input with an existed data file.
    example_input_path = os.path.join('path', 'to', 'input_data')
    dp1.add_input_by_data_path(example_input_path)

    # Add output.
    # data_type: 0 (float32), 1 (float16)
    size = 2
    data_type = 1
    dp1.add_output('out0.data', size, data_type)

    # Add expect data.
    example_expect_data = bytes([0x3f, 0x1b, 0xf2, 0xcf])
    dp1.add_expect_by_data(example_expect_data)

    # Or add expect data file path.
    example_expect_path = os.path.join('path', 'to', 'expect_data')
    dp1.add_expect_by_data_path(example_expect_path)

    # [OPTIONAL] Set precision_deviation and statistical_discrepancy.
    dp1.set_precision_deviation(0.2)
    dp1.set_statistical_discrepancy(0.2)

    return dp1

def testcase():
    # Write your code here. The function 'example' can be a reference.
    # You must RETURN an instance of DataProvider.
    return example()
