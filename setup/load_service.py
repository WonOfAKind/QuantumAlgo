from qiskit_ibm_runtime import QiskitRuntimeService

# Load the service
service = QiskitRuntimeService()
print(service.backends())
