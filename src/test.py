from Model import Model

example_parameters = [7.38, 2.2456, -10.10, 1.15, 8572.40, 4380.50, 11510.50, 2155.30, 98.50, 1025.30, 612.30, 11820.40, 2185.30]
example_2 = [7.38, 2245.60, -10.10, 1.15, 8572., 4380.50, 11510.50, 2155.30, 98.50, 1025.30, 612.30, 11820.40, 2185.30]

try:
    model_instance = Model()
    result = model_instance.predict(example_2)
    print(result)
except Exception as e:
    print("Error", e)
