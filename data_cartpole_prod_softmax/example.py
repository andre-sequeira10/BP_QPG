import pennylane as qml 
from pennylane import numpy as np

n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(w,local=True,seed=None):
    
    qml.RandomLayers(weights=w, wires=range(n_qubits), seed=seed)

    if local:
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, n_qubits-1])

        return qml.expval(qml.PauliZ(n_qubits-1))

    else:
        return qml.expval(qml.PauliZ(0)@qml.PauliZ(1)@qml.PauliZ(2)@qml.PauliZ(3)@qml.PauliZ(4))


for i in range(100):
    shape = qml.RandomLayers.shape(n_layers=3, n_rotations=30)
    w = np.random.random(size=shape)
    print("TEST {} - {} || {} ".format(i,circuit(w,local=True,seed=9),circuit(w,local=False,seed=9)))
