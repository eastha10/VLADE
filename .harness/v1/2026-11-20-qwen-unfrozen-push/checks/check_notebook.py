"""Execute notebook planning and command cells with isolated, simulated runtimes."""
import ast
import copy
import json
from pathlib import Path
import sys
import nbformat

notebook_path, scratch = map(Path,sys.argv[1:])
notebook=nbformat.read(notebook_path,as_version=4)
nbformat.validate(notebook)
cells={cell.id:cell.source for cell in notebook.cells}
for cell in notebook.cells:
    if cell.cell_type=='code':
        compile(cell.source,cell.id,'exec')
        assert not cell.outputs and cell.execution_count is None
assert len(cells)==len(notebook.cells)
tree=ast.parse(cells['settings'])
for node in tree.body:
    if isinstance(node,ast.Assign):
        names=[t.id for t in node.targets if isinstance(t,ast.Name)]
        if 'WORK' in names:node.value=ast.parse('Path('+repr(str(scratch.resolve()))+')',mode='eval').body
        if 'VARIANTS' in names:node.value=ast.Constant('both')
namespace={}
exec(compile(ast.fix_missing_locations(tree),'settings','exec'),namespace)
namespace['SOURCE_COMMIT']='a'*40
exec(cells['run-contract'],namespace)
assert namespace['COMMON']['trainable_scope']=='full'
assert namespace['COMMON']['top_layer_count']==0
assert namespace['SETTINGS']['gt_only']=={k:v for k,v in namespace['SETTINGS']['gt_kd'].items() if k!='distillation'}
exec(cells['run-contract'],namespace)
namespace['MAX_STEPS']+=1
try:exec(cells['run-contract'],namespace)
except ValueError as error:assert '기존 실행' in str(error)
else:raise AssertionError('Accepted changed run settings')
namespace['MAX_STEPS']-=1
exec(cells['run-contract'],namespace)
namespace.update(CODE=Path('git-source'),ENV={},STUDENT_PYTHON=Path('student-python'),TEACHER_PYTHON=Path('teacher-python'))
exec(cells['runner'],namespace)
commands=[]
def invoke(command,env):
    command=[str(value) for value in command]
    commands.append((command,copy.deepcopy(env)))
    if '--stop-after' in command:
        output=Path(command[command.index('--output')+1])
        step=int(command[command.index('--stop-after')+1])
        checkpoint=output/f'step-{step:07d}'
        checkpoint.mkdir(parents=True)
        (checkpoint/'COMPLETE').touch()
namespace['invoke']=invoke
try:exec(cells['train'],namespace)
except RuntimeError:pass
else:raise AssertionError('Main training accepted without a pilot')
exec(cells['pilot'],namespace)
assert len(commands)==2
assert '--teacher-python' not in commands[0][0]
assert commands[1][0][-2:]==['--teacher-python','teacher-python']
assert all('--resume' not in row[0] for row in commands)
assert commands[0][1]['VLADE_HUB_RUN_ID']!=commands[1][1]['VLADE_HUB_RUN_ID']
exec(cells['train'],namespace)
assert len(commands)==4 and all('--resume' in row[0] for row in commands[-2:])
exec(cells['train'],namespace)
assert all('-c' in row[0] for row in commands[-2:])  # completed checkpoints are verified before skip
assert 'configure_trainable_scope' in cells['source'] and 'RunTimings' in cells['source']
assert 'resolve_input' in cells['preflight'] and 'hub_checkpoints.py' in cells['preflight']
assert "TRAINABLE_SCOPE = \"full\"" in cells['settings']
assert 'top_layer_count' in cells['run-contract']
print(json.dumps({'notebook_checks':'passed','cells':len(cells),
    'default_full':True,'matched_comparison':True,'pilot_and_resume_commands':True,
    'changed_contract_refused':True,'completed_checkpoint_verification':True}))
