"""Compare private dev6 to untouched dev5, ignoring only approved namespace edits."""
from pathlib import Path
import argparse,ast,hashlib,json
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-root',type=Path,required=True,help='Preserved dev5 src directory')
args=parser.parse_args()
OLD=args.baseline_root.resolve()
NEW=R/'src'
manifest=json.loads((R/'SOURCE_MANIFEST.json').read_text())
paths=json.loads((R/'verification/namespace/migration-map.json').read_text())
reverse={'.optimization':'ubukit.optimization','._rough_memberships':'ubukit._rough_memberships','._impl.portable_accel':'portable_accel','._impl.fcm':'ubukit_fcm','._impl.rmcm':'ubukit_rmcm','._impl.rough_cmeans':'rough_cmeans','._impl.external_metrics':'external_metrics'}
class Normalize(ast.NodeTransformer):
 def __init__(self,facade):self.facade=facade
 def visit_Import(self,node):return None
 def visit_ImportFrom(self,node):return None
 def generic_visit(self,node):
  node=super().generic_visit(node)
  if hasattr(node,'body') and isinstance(node.body,list) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
  return node
 def visit_Constant(self,node):
  if self.facade and isinstance(node.value,str):
   if node.value==manifest['version']:node.value=manifest['baseline_version']
   elif node.value in reverse:node.value=reverse[node.value]
  return node
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if self.facade and isinstance(node.func,ast.Name) and node.func.id=='import_module' and len(node.args)==2 and isinstance(node.args[1],ast.Name) and node.args[1].id=='__name__':node.args=node.args[:1]
  return node
rows=[]
for old,new in sorted(paths.items()):
 a=(OLD/old).read_bytes();b=(NEW/new).read_bytes();same=a==b
 norm=lambda data:ast.dump(Normalize(old=='ubukit/__init__.py').visit(ast.parse(data)),include_attributes=False)
 assert norm(a)==norm(b),old
 rows.append({'path':'src/'+new,'baseline_path':'src/'+old,'sha256':hashlib.sha256(b).hexdigest(),'baseline_sha256':hashlib.sha256(a).hexdigest(),'change':'unchanged_bytes' if same else 'namespace_import_facade_or_doc_only','numerical_ast_equal':True})
extra=NEW/'ubukit/_impl/__init__.py';rows.append({'path':'src/ubukit/_impl/__init__.py','sha256':hashlib.sha256(extra.read_bytes()).hexdigest(),'change':'new_private_package_marker'})
assert len(rows)==len(manifest['files'])==len(paths)+1
assert {p.relative_to(NEW).as_posix() for p in NEW.rglob('*.py')}=={r['path'][4:] for r in rows}
changed=[r['baseline_path'] for r in rows if r['change']=='namespace_import_facade_or_doc_only']
assert set(changed)=={'src/ubukit/__init__.py','src/ubukit/_rough_memberships.py','src/rough_cmeans.py','src/external_metrics.py'},changed
m={**manifest,'files':rows}
assert m==json.loads((R/'SOURCE_MANIFEST.json').read_text()), 'source manifest mismatch'
r={'status':'passed','inherited_runtime_files':len(paths),'runtime_files':len(rows),'byte_identical_inherited_files':sum(r['change']=='unchanged_bytes' for r in rows),'normalized_numerical_ast_equal_files':len(paths),'changed_inherited_files':changed,'new_file':'src/ubukit/_impl/__init__.py','normalization_scope':['Import/ImportFrom syntax','docstrings','exact facade module-target strings','facade version dev6 to dev5','relative import_module package argument in facade'],'no_numerical_parameter_expression_or_data_edits':True}
print(json.dumps(r,indent=2))
