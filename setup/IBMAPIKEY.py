from qiskit_ibm_runtime import QiskitRuntimeService
import os

token = os.environ.get('token')
crn = os.environ.get('crn')

# Replace with your actual IBM API Token
service = QiskitRuntimeService.save_account(
    token=token,
    instance=crn,  # Replace with your IBM Cloud CRN
    name="Wontum",       # Optional: give it a name
    set_as_default=True          # Set as default to avoid specifying it each time
)
