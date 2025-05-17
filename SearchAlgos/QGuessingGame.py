import random
import math
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile

# Oracle for a single marked state
def oracle(qc: QuantumCircuit, qr: QuantumRegister, target: int):
    """Flip the sign of the target state."""

    num_qubits = len(qr)
    # Apply X gates to match the target state to |111...1>
    for i in range(num_qubits):
        if ((target >> i) & 1) == 0:
            qc.x(qr[i])
    # Multi-controlled Z (as H-X-mcx-X-H)
    qc.h(qr[-1])
    qc.mcx(qr[:-1], qr[-1])
    qc.h(qr[-1])
    # Undo X gates
    for i in range(num_qubits):
        if ((target >> i) & 1) == 0:
            qc.x(qr[i])

# Diffusion operator
def diffusion(qc: QuantumCircuit, qr: QuantumRegister):
    """Grover's diffusion (inversion about the mean) operator."""

    num_qubits = len(qr)
    qc.h(qr)
    qc.x(qr)
    qc.h(qr[-1])
    qc.mcx(qr[:-1], qr[-1])
    qc.h(qr[-1])
    qc.x(qr)
    qc.h(qr)

def groversAlgorithm(target: int, num_qubits: int, shots: int = 1024):
    """
    Runs Grover's Algorithm for a single marked element using external oracle/diffusion functions.
    If use_ibm_runtime=True, will submit to IBM Quantum hardware or their cloud simulator.
    Returns: (most_measured_result, counts, num_iterations)
    """
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qr, cr)

    # Step 1: Superposition
    qc.h(qr)

    # Step 2: Grover iterations
    N = 2 ** num_qubits
    num_iterations = int((math.pi/4) * math.sqrt(N))
    for _ in range(num_iterations):
        oracle(qc, qr, target)
        diffusion(qc, qr)

    # Step 3: Measurement
    qc.measure(qr, cr)

    # Initialize the Qiskit Runtime Service
    service = QiskitRuntimeService()

    # Select an appropriate backend
    backend = service.least_busy(simulator=False, operational=True)

    # Transpile the circuit for the selected backend
    compiled_circuit = transpile(qc, backend)

    # Initialize the Sampler primitive
    sampler = Sampler(mode=backend)

    # Run the sampler
    result = sampler.run([compiled_circuit], shots=shots).result()
    counts = result[0].join_data().get_counts()

    # Process the results
    most_likely = max(counts, key=counts.get)
    measured_int = int(most_likely, 2)

    return measured_int, counts, num_iterations

def linearSearchAlgorithm(target: int, arr: list) -> int:
    """Return the number of steps a naive linear scan needs to hit the target."""
    num_steps = 0
    for num in arr:
        if num == target:
            return num_steps + 1
        num_steps += 1
    return num_steps

def main():
    #Some Command Line GUI thingy stuff
    while True:
        print(" === Welcome to the Quantum Guessing Game! === ")
        print("Game modes:\n"
              "1. Compare classical vs. quantum search algorithm\n"
              "2. Play the guessing game with a random number!\n"
              "3. Exit the game\n")

        #Select the Gamemode
        game_mode = input("Make your selection: ")

        #Select
        if game_mode == "1":
            print("Let the battle begin!")
            try:
                qubit = int(input("Enter the amount of qubits between 1 and 20: "))
            except ValueError:
                print("Please enter a number!")
                continue

            #One qubit can either be a 0 or 1. For every qubit added, the amount of numbers that can be represented goes up
            #2 ^ n, where n is the amount of qubits. For example, we can represent 2 qubits as 00, 01, 10, 11.
            rang = 2 ** qubit
            print('Type "1" if you want to choose the number. Type "2" if you want it to be randomly selected.')
            random_choice = input("Select your option: ")

            if random_choice == "1":
                target_num = int(input(f"Enter the target number between 0 and {rang - 1}: "))
            elif random_choice == "2":
                target_num = random.randint(0, rang - 1)

            shots = int(input("How many times do you want to run Grover's Algorithm? (Higher shot count will increase accuracy): "))
            arr = list(range(0, rang))
            random.shuffle(arr)
            print(arr)

            classical_steps = linearSearchAlgorithm(target_num, arr)
            print(f"Linear search needed {classical_steps} checks to find our target number.")

            measured_number, counts, quantum_steps = groversAlgorithm(target_num, qubit, shots)
            accuracy = (counts[str(bin(target_num)[2:])] / shots) * 100

            print(f"The quantum algorithm measured {measured_number}.\n"
                  f"It took {quantum_steps} steps to find the target number.\n"
                  f"The accuracy of Grover's Algorithm is {accuracy:.2f}%. ")


        if game_mode == "2":
            print("Time to guess!")

            max_range = int(input("Enter the highest number that can be chosen to be guessed: "))
            print("Picking random number...")

            random_number = random.randint(1, max_range)
            attempts = 1
            guessing_game = True

            while guessing_game:

                guess = int(input(f"Guess a number between 1 and {max_range}: "))
                if random_number == guess:
                    print(f"Correct! it took you {attempts} attempts.")
                    guessing_game = False
                elif random_number < guess:
                    print("Incorrect! try picking a lower number.")
                elif random_number > guess:
                    print("Incorrect! try picking a higher number.")

                attempts += 1

        if game_mode == "3":
            print("Thank you for playing! Goodbye!")
            break


if __name__ == "__main__":
    main()