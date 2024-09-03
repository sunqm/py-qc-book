import math
import itertools
from typing import List

VSYMBOLS = 'abcdefghvwxyzABCDEFGHVWXYZ'
OSYMBOLS = 'ijklmnopqrstuIJKLMNOPQRSTU'
COMMUTABLE = {'V': 'O', 'O': 'V', '*': ''}

class FieldOperator:
    '''Annihilation or creation operators'''
    def __init__(self, label='*'):
        assert label in 'OV*'
        self.label = label

class Annihilation(FieldOperator):
    def __repr__(self):
        return f'{self.label}'

class Creation(FieldOperator):
    def __repr__(self):
        return f'{self.label}^+'

class Delta:
    '''the Kronecker delta.'''
    def __init__(self, indices: List[FieldOperator]):
        self.indices = indices

class Tensor:
    def __init__(self, indices: List[FieldOperator], label=None):
        self.indices = indices
        self.label = label

    def __repr__(self):
        ops = ' '.join(op.label for op in self.indices)
        return f'{self.label or self.__clas__.__name__}({ops})'

class String:
    '''
    Normal-ordered FieldOpeartors
    '''
    __slots__ = ['operators', 'factor', 'deltas', 'remaining']

    def __init__(self, operators, factor=1, deltas=None, remaining=None):
        self.operators: List[FieldOperator] = operators
        self.factor = factor
        self.deltas = deltas or []
        self.remaining: List[FieldOperator] = remaining or []

    def __repr__(self):
        ops = ' '.join(repr(op) for op in self.operators + self.deltas + self.remaining)
        return f'{self.__class__.__name__}({ops}) '

def commutable(op1: FieldOperator, op2: FieldOperator):
    return op1.__class__ == op2.__class__ or op1.label == COMMUTABLE[op2.label]

def _contract_fop_fop(fop1: FieldOperator, fop2: FieldOperator) -> List[String]:
    if not commutable(fop1, fop2):
        yield String([], 1, [Delta([fop1, fop2])])
    yield String([fop2], -1, remaining=[fop1])

def _contract_fop_string(fop: FieldOperator, string: String) -> List[String]:
    operators = string.operators
    remaining = string.remaining
    factor = string.factor
    deltas = string.deltas
    if len(operators) == 0:
        yield String(operators, factor, deltas, [fop]+remaining)
        return

    # C(x, a1*a2*...) = C(x, a1)*a2*a3*...
    #                 = delta(x,a1)*a2*a3*... + C(remaining, a2*a3*...)
    for s in _contract_fop_fop(fop, operators[0]):
        if not s.remaining:
            yield String(operators[1:], s.factor * factor,
                         s.deltas + deltas, remaining)
        else:
            substring = String(operators[1:], factor, deltas, remaining)
            for r in _contract_fop_string(s.remaining[0], substring):
                yield String(s.operators + r.operators, s.factor * r.factor,
                             r.deltas, r.remaining)

def _contract_string_string(string1, string2) -> List[String]:
    assert isinstance(string1, String) and not string1.remaining
    assert isinstance(string2, String) and not string2.remaining
    if not string2.operators:
        return [String(string1.operators, string1.factor * string2.factor,
                       string1.deltas + string2.deltas)]

    output = [String(string2.operators, string1.factor * string2.factor,
                     string1.deltas + string2.deltas)]
    for fop in reversed(string1.operators):
        # C(x1*x2*..., a1*a2*...) = C(x1, C(x2, ..., C(xn, a1*a2*...)))
        output = flatten(_contract_fop_string(fop, s) for s in output)
    return output

def contract(string1, string2):
    return (String(s.operators + s.remaining, s.factor, s.deltas)
            for s in _contract_string_string(string1, string2))

def commutate(string1, string2) -> List[String]:
    '''Simplifies the commutator [string1, string2]'''
    output = contract(string1, string2)
    n_deltas = len(string1.deltas) + len(string2.deltas)
    return (s for s in output if len(s.deltas) > n_deltas)

def flatten(lst):
    return itertools.chain(*lst)

def on_HF_reference(string, tensors, output_indices=None) -> List[str]:
    '''<HF|string|HF>'''
    h_ops = []
    for op in string.operators:
        if op.label == '*':
            h_ops.append(op)
            continue
        # Creation or annihilation operators in T are normal ordered.
        # They must be zero when applied either on |HF> or on <HF|
        return ''

    deltas = string.deltas
    if h_ops:
        # The remaining operators in Hamiltonian
        if (len(h_ops) == 2 and
            isinstance(h_ops[0], Creation) and isinstance(h_ops[1], Annihilation)):
            deltas = deltas + [Delta(h_ops)]
        elif len(h_ops) == 4:
            return f'{string.factor*2} * einsum("ijij->", h2_oooo)'
        else:
            return ''

    vsymbols, osymbols = iter(VSYMBOLS),iter(OSYMBOLS)
    symbol_table = {}
    for delta in deltas:
        op1, op2 = delta.indices
        label1, label2 = op1.label, op2.label
        if label1 == 'V' or label2 == 'V':
            symbol_table[op1] = symbol_table[op2] = next(vsymbols)
        else:
            symbol_table[op1] = symbol_table[op2] = next(osymbols)

    if output_indices is None:
        output_script = ''
    else:
        output_script = ''.join(symbol_table[i] for i in output_indices)
    scripts = []
    operands = []
    for tensor in tensors:
        s = ''.join(symbol_table[i] for i in tensor.indices)
        scripts.append(s)
        if tensor.label[0] == 'h':
            operands.append(
                tensor.label + '_' + ''.join('v' if x in VSYMBOLS else 'o' for x in s))
        else:
            operands.append(tensor.label)
    scripts = ','.join(scripts)
    operands = ', '.join(operands)
    return f'{string.factor} * einsum("{scripts}->{output_script}", {operands})'

def _make_operator(factor, indices, label=None):
    n = len(indices)
    assert n % 2 == 0
    operators_c = [Creation(x) for x in indices[:n//2].upper()]
    operators_a = [Annihilation(x) for x in indices[n//2:].upper()]
    operators = operators_c + operators_a[::-1]
    tensor = Tensor(operators_c + operators_a, label)
    return tensor, String(operators, factor)

def hamiltonian(n, factor):
    return _make_operator(factor, '**'*n, f'h{n}')

def excitation(n):
    return _make_operator(.25, 'V'*n + 'O'*n, f't{n}')

def deexcitation(n):
    return _make_operator(.25, 'O'*n + 'V'*n, f't{n}')

def bra_projector(n):
    bra_tensor, bra = _make_operator(.25, 'O'*n + 'V'*n)
    return bra_tensor.indices[::-1], bra

def apply_BCH(truncation=4):
    hs = [hamiltonian(1, 1), hamiltonian(2, .25)]
    for order in range(truncation+1):
        factor = 1./math.factorial(order)
        for htensor, hstring in hs:
            ht_tensors = [htensor]
            ht_strings = [String(hstring.operators, hstring.factor * factor)]
            for i in range(order):
                amplitudes, t = excitation(2)
                ht_tensors.append(amplitudes)
                ht_strings = flatten(commutate(ht, t) for ht in ht_strings)
            for ht_string in ht_strings:
                yield ht_tensors, ht_string

def CCD_energy(truncation=4):
    output = []
    for ht_tensors, ht_string in apply_BCH(truncation):
        output.append(on_HF_reference(ht_string, ht_tensors))
    return [x for x in output if x]

def CCD_equations(truncation=4):
    output_indices, bra = bra_projector(2)
    output = []
    for ht_tensors, ht_string in apply_BCH(truncation):
        for s in contract(bra, ht_string):
            output.append(on_HF_reference(s, ht_tensors, output_indices))
    return [x for x in output if x]

if __name__ == '__main__':
    for x in CCD_energy(2):
        print(f'E += {x}')

    for x in CCD_equations(2):
        print(f'F2 += {x}')
