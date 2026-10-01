#!/usr/bin/env python3
"""Build the restart-v2 report from current verified normalized JSON data.
Historical reports remain separate. The report itself does not rerun benchmarks.
"""
from pathlib import Path
from xml.sax.saxutils import escape
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont as FTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
import hashlib,json
HERE=Path(__file__).resolve().parent
OUT=HERE/'output/pdf'; TMP=HERE/'tmp/pdfs'
OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
DATA=json.loads((HERE/'report_data.json').read_text())
assert DATA['final_metrics_verified'] is True
kr=DATA['kmeans']; sr=DATA['som']; mr=DATA['metrics']
copy={
 'title':'高速化チャレンジ restart v2 結果要約',
 'date':DATA['as_of_label'],
 'lead':f"共通の公開APIで、SOM-OLPは{sr['ratio']:.2f}倍、T/C同時計算は最大{mr['best_ratio']:.2f}倍。k-meansは最終出力まで含めて低次元{kr['low']['ratio']:.2f}倍 / 高次元{kr['high']['ratio']:.2f}倍でした。",
 'scope':'各課題・各条件の中だけで比較した代表候補の順位です。旧実験の数値は今回値に流用していません。',
 'km_title':'1  k-means',
 'km_sub':'同じ初期中心、20反復、9スレッド。公開APIの最終割当・inertia計算まで包含 [1]',
 'km_note':f"3方式とも最終割当・inertia込み。NumPy参照（特徴順の直接距離、B256）比は約{kr['low']['numpy_ratio']:.1f}/{kr['high']['numpy_ratio']:.1f}倍で、最速NumPy比ではありません。最終ラベルは完全一致、中心・inertiaは許容差内。時間には変動があり、先行窓2.12/1.21倍[6]と混ぜず最新窓を採用。直接最終割当は明示選択で、汎用sklearn互換の保証ではありません。",
 'som_title':'2  SOM-OLP',
 'som_sub':'MNIST全70,000件 × 784次元、256ノード。初期化から収束36反復までの公開API [2]',
 'som_note':'完全SVDを保ち、初期確率の低ランク式とThreadPoolを明示選択。入力検証・初期化・全反復・出力生成を計時内に含めました。初期化は両者9スレッド、最適化学習部は9ワーカー＋BLAS 1スレッドです。別枠のnative 2.581秒は固定10反復・初期化除外のカーネル値で、上表と混ぜません。',
 'metric_title':'3  Trustworthiness と Continuity の同時計算',
 'metric_sub':mr['subtitle']+' [3]',
 'metric_note':mr['note'],
 'timing':'測定条件  Linux x86_64 / AMD EPYC 9V74 / CPU 9スレッド / GPUなし。k-meansは7回、他は3回のウォーム中央値。初回は別記録、import・JIT準備・結果の数値検証は計時外です。単一環境・選定データの結果で、速度の一般保証ではありません。',
 'p2_title':'OSと配布のしやすさ',
 'p2_lead':'共通Pythonパッケージを土台にし、必要な場所だけNumbaやThreadPoolを明示選択する構成です。速い設定が既定設定とは限らず、用途ごとの確認が必要です。',
 'os_caveat':'実機検証はすべてLinux x86_64のみです。上流ライブラリの対応と、この試作の動作確認は別です。Intel MacはNumbaのTier 1対象外で、0.63以降はTier 2bのコミュニティ対応です。NumPy経路を優先候補にします [7]。',
 'quality_title':'正確さと検証範囲',
 'quality1':'SOM-OLPはW・P・V・全目的関数履歴が元実装とrtol=atol=1e-8で一致し、反復数は同じ36回。各設定内の反復結果はビット単位で安定しましたが、元実装とのビット一致を意味しません。今回の全70k埋め込みについて、T/C整数ペナルティの再比較までは実施していません。',
 'quality2':DATA['verification_note'],
 'reuse_title':'共通化と実用上の注意',
 'reuse':'別条件のPreparedSOMでは、同じ70k入力でγ/λを変える3 fitが40.008→33.089秒（1.21倍、毎回の準備4.189秒込み）。出力・履歴はfreshと完全一致し、追加の所有配列は約420 MiB。1 fitでは遅く、この条件では2 fit以降に利益が出ました [5]。',
 'research':'追加検証では1スレッドの低次元k-meansはsklearnより遅く、Digitsではsqrt版より通常Numba版が高速でした。常に同じ方式が最速とは限りません。再利用の大型計時は独立候補で実施し、a2 API統合後は小規模一致を確認。主表の12.90倍と掛け合わせません。T/Cの距離メモリはO(N²)、SOMの64 MiB作業枠はRSS上限ではありません。BLASのスレッド制限はプロセス全体に作用します。',
 'env':'検証環境  Python 3.12.14 / NumPy 2.3.5 / SciPy 1.17.0 / scikit-learn 1.8.0 / Numba 0.67.0。一般公開・本番運用・他OSでの実測は未実施です。',
 'sources_title':'今回の検証記録',
 'recovery':'再現用のソースパッケージは別添です。入力データ本体・ビルド済みネイティブ・JITキャッシュは含めず、入力の由来・生成手順・ハッシュを残しています。これは新たな再構築版の成果で、旧版の同一バイト復旧ではありません。',
}
km=[['条件','順位','実装','時間','対sklearn'],['20,000 × 8\nK=16','1','公開Numba ＋ 直接最終割当',f"{kr['low']['candidate_s']*1000:.3f} ms",f"{kr['low']['ratio']:.2f}倍"],['','2','sklearn Lloyd',f"{kr['low']['baseline_s']*1000:.3f} ms",'基準'],['','3','自作NumPy参照 ＋ 最終割当',f"{kr['low']['numpy_s']*1000:.3f} ms",'参照'],['10,000 × 128\nK=64','1','公開Numba ベクトルBLAS ＋ 最終割当',f"{kr['high']['candidate_s']*1000:.3f} ms",f"{kr['high']['ratio']:.2f}倍"],['','2','sklearn Lloyd',f"{kr['high']['baseline_s']*1000:.3f} ms",'基準'],['','3','自作NumPy参照 ＋ 最終割当',f"{kr['high']['numpy_s']:.3f} s",'参照']]
som=[['順位','実装','時間','対元実装'],['1','公開ThreadPool ＋ 低ランク初期確率式',f"{sr['candidate_s']:.3f} s",f"{sr['ratio']:.2f}倍"],['2','元のSOMOLP.fit',f"{sr['baseline_s']:.3f} s",'基準']]
met=[['順位','実装','時間','対sklearn 2回']]+[[str(i+1),r['label'],f"{r['seconds']:.3f} s",'基準' if r['baseline'] else f"{r['ratio']:.2f}倍"] for i,r in enumerate(mr['rows'])]
osrows=[['配布容易性','構成と導入負担','OSとCPUの位置づけ'],['1位\n共通 標準依存版','NumPy / SciPy / sklearn等。独自C/C++コンパイラ・Cython不要。SOM ThreadPoolもNumba不要。','Linux x86_64は実測済み。Windows・macOS・ARMは移植候補で未検証。依存ライブラリの対応配布物が必要。'],['2位\n共通 ＋ Numba','追加依存はNumba / llvmlite。初回JIT時間が必要。対応wheel利用時は独自ビルド不要。','上流Tier 1はWindows x64、Linux x64 / ARM64、macOS Apple Silicon。この試作の実機確認はLinux x64のみ [7]。'],['3位\n専用 Cython','GCC・OpenMP・libmvec等、CPU/OS依存のビルドが必要。標準パッケージの必須依存にはしない。','今回のLinux x86_64構成でのみ確認。Windows・macOS・ARMへのビルド移植・検証、汎用native wheelは未整備。']]
sources=['[1] restart_v2/evidence/kmeans_scratch_matched/{low_d,high_d,summary}.json','[2] restart_v2/som_candidate/results/fullfit70k/{original,portable}_public_comparison.json','[3] '+mr['source_label'],'[4] restart_v2/README.md、evidence/ 内の配布QA記録、requirements-{foundation,tested}.txt','[5] restart_v2/som_candidate/results/prepared_sweep70k/{fresh,prepared}.json','[6] restart_v2/kmeans_candidate/results/matched_v2/{low_d,high_d,summary}.json','[7] Numba公式 Support Policy / Installation  2026年9月30日確認']
text='\n'.join(copy.values())+'\n'.join(sources)+'\n'.join(str(c) for rows in [km,som,met,osrows] for row in rows for c in row)+'0123456789 / 研究用試作 実測と検証の要約 restart-v2 新規測定値'
def font(weight):
 src=FTFont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+weight+'.ttc',fontNumber=0);cm=src.getBestCmap();wanted=set(text)|set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:/._{}[]+-<>%=(),');missing=[c for c in wanted if ord(c) not in cm and not c.isspace()];assert not missing,missing
 cmap={ord(c):cm[ord(c)] for c in wanted if ord(c) in cm};order=['.notdef']+sorted(set(cmap.values())-{'.notdef'});gs=src.getGlyphSet();glyphs={}
 for name in order:
  pen=TTGlyphPen(None);gs[name].draw(Cu2QuPen(pen,max_err=1.0,reverse_direction=True));glyphs[name]=pen.glyph()
 fb=FontBuilder(src['head'].unitsPerEm,isTTF=True);fb.setupGlyphOrder(order);fb.setupCharacterMap(cmap);fb.setupGlyf(glyphs);fb.setupHorizontalMetrics({g:src['hmtx'][g] for g in order});fb.setupHorizontalHeader(ascent=src['hhea'].ascent,descent=src['hhea'].descent);fb.setupNameTable({'familyName':'Noto Sans JP Report','styleName':weight,'uniqueFontIdentifier':'NotoSansJPReport-'+weight,'fullName':'Noto Sans JP Report '+weight,'psName':'NotoSansJPReport-'+weight});fb.setupOS2(sTypoAscender=880,sTypoDescender=-120,usWinAscent=1160,usWinDescent=288);fb.setupPost();fb.setupMaxp();fp=TMP/('NotoSansJP-'+weight+'.ttf');fb.save(fp);pdfmetrics.registerFont(TTFont('JP-'+weight,str(fp)))
font('Regular');font('Bold');pdfmetrics.registerFontFamily('JP-Regular',normal='JP-Regular',bold='JP-Bold')
W,H=A4;M=40;WIDTH=W-2*M
styles={}
def st(name,size,lead,bold=False,after=0,before=0,color=colors.black):
 styles[name]=ParagraphStyle(name,fontName='JP-Bold' if bold else 'JP-Regular',fontSize=size,leading=lead,spaceAfter=after,spaceBefore=before,textColor=color,wordWrap='CJK')
st('title',19,25,True,6);st('date',8.8,12,after=12,color=colors.HexColor('#535962'));st('lead',10.8,16.2,True,7);st('body',9.2,13.7,after=6);st('note',8.2,12,after=7);st('small',7.7,10.7,after=4);st('h2',12.5,17.2,True,5,9);st('sub',8.4,11.7,after=5);st('cell',8.7,12.1);st('cellsmall',8.5,12.4);st('tablehead',8.5,11.8,True,color=colors.white);st('source',7.1,10.5,after=2)
def p(t,s='body',raw=False):return Paragraph(t if raw else escape(t).replace('\n','<br/>'),styles[s])
def table(rows,widths,numeric=(),spans=(),small=False):
 arr=[]
 for i,row in enumerate(rows):
  line=[]
  for j,c in enumerate(row):
   style=styles['tablehead' if i==0 else 'cellsmall' if small else 'cell']
   if j in numeric:style=ParagraphStyle(f'{style.name}-{i}-{j}',parent=style,alignment=TA_CENTER)
   line.append(Paragraph(escape(str(c)).replace('\n','<br/>'),style))
  arr.append(line)
 t=Table(arr,colWidths=widths,hAlign='LEFT');cmd=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#243447')),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5.4),('BOTTOMPADDING',(0,0),(-1,-1),5.4)]
 for i in range(2,len(rows),2):cmd.append(('BACKGROUND',(0,i),(-1,i),colors.HexColor('#F4F6F8')))
 for a,b in spans:cmd.append(('SPAN',a,b))
 t.setStyle(TableStyle(cmd));return t
story=[p(copy['title'],'title'),p(copy['date'],'date'),p(copy['lead'],'lead'),p(copy['scope'],'note')]
story += [p(copy['km_title'],'h2'),p(copy['km_sub'],'sub'),table(km,[91,34,211,94,WIDTH-430],(1,3,4),(((0,1),(0,3)),((0,4),(0,6)))),Spacer(1,6),p(copy['km_note'],'note')]
story += [p(copy['som_title'],'h2'),p(copy['som_sub'],'sub'),table(som,[39,292,100,WIDTH-431],(0,2,3)),Spacer(1,6),p(copy['som_note'],'note')]
story += [p(copy['metric_title'],'h2'),p(copy['metric_sub'],'sub'),table(met,[39,282,97,WIDTH-418],(0,2,3)),Spacer(1,6),p(copy['metric_note'],'note'),Spacer(1,4),p(copy['timing'],'small'),PageBreak()]
story += [p(copy['p2_title'],'title'),Spacer(1,4),p(copy['p2_lead']),Spacer(1,3),table(osrows,[95,187,WIDTH-282],small=True),Spacer(1,7),p(copy['os_caveat'],'note'),p(copy['quality_title'],'h2'),p(copy['quality1']),p(copy['quality2']),p(copy['reuse_title'],'h2'),p(copy['reuse']),p(copy['research']),Spacer(1,5),p(copy['env'],'small'),p(copy['sources_title'],'h2')]
for x in sources[:-1]:story.append(p(x,'source'))
story.append(p('[7] Numba公式 <a href="https://numba.readthedocs.io/en/stable/reference/support_tiers.html" color="#17496E">Support Policy</a> / <a href="https://numba.readthedocs.io/en/stable/user/installing.html" color="#17496E">Installation</a>  2026年9月30日確認','source',True))
story += [Spacer(1,7),p(copy['recovery'],'small')]
def footer(c,d):
 c.saveState();c.setStrokeColor(colors.HexColor('#D9D9D9'));c.setLineWidth(.45);c.line(M,31,W-M,31);c.setFillColor(colors.HexColor('#59616C'));c.setFont('JP-Regular',7.3);c.drawString(M,20,'研究用試作  restart-v2 新規測定値');c.drawRightString(W-M,20,f'{d.page} / 2');c.restoreState()
pdf=OUT/'restart_v2_final_report_ja.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=36,bottomMargin=44,title=copy['title'],author='dot',subject='restart-v2のk-means SOM-OLP T/C新規測定と配布性の要約')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(HERE/'report_build_record.json').write_text(json.dumps({'status':'Built from current verified restart-v2 records','pdf':str(pdf),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'source_labels':sources},ensure_ascii=False,indent=2))
print(pdf)
