"""Freeze the weak-encoder ablation (EAB round 12, reviewer W2).

Question: is "acquisition rights decide private value" (Finding 1) specific to the
DINOv2 representation? The study replaces the frozen public encoder by an
ImageNet ResNet-18 and reruns the CIFAR-1k independent-confirmation comparison
with everything else fixed: same image identities and splits, same proxy-fitted
128-d PCA and row normalization, same client populations (cohort seeds
8801-8810 regenerate identical client memberships), same 180 private steps,
same K=C regional release at epsilon=.1, same independent random label draws
(confirmation/samples, draws 0-99), same fourteen public learners, joint
lambdas and exact LOO selection (release-frontier anchored_fusion, unchanged).
"""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def design():
    return dict(
        stage='EAB round 12: weak-encoder ablation of Finding 1',
        encoder=dict(name='torchvision ResNet-18, IMAGENET1K_V1 weights (downloaded with user approval 2026-10-22)',
            layer='global average pool, 512-d, eval mode, frozen',preprocessing='as DINOv2: bicubic resize to 224, ImageNet mean/std',
            reduction='PCA to 128 fitted on the 2000 public proxy images only (randomized, random_state 1729), then row L2 normalization'),
        reference_encoder='DINOv2 ViT-S/14 (all existing results)',
        identities='same private/development/proxy/validation/test image ids as release-frontier inputs/cifar10.npz; labels asserted equal',
        setting=dict(dataset='cifar10',clients=1000,dirichlet_alpha=.1,epsilon=.1,delta=1e-5,private_steps=180,release='T=1, K=C=10 public regions, estimator M',
            cohort_seeds=list(range(8801,8811)),head_fixed_release='also run (clip 1), as in the reference confirmation'),
        labels=dict(random='release-frontier confirmation/samples cifar10_random_{0..99}_m{10,30,50,100,200,500}.json, unchanged indices',
            feature_directed='geometry_labels policy (q nearest development points per public K=C region, recomputed in the ResNet-18 geometry), q in {1,3,5}'),
        evaluation='release-frontier anchored_fusion.main(confirm=True) with module roots pointed at this stage; joint naming and fourteen public learners with exact LOO',
        noise_note='release noise keys are the reference formula (seed*1000003+steps*101+20261019), so the two encoders share noise tapes; paired by design',
        primary=dict(
            E1=('random acquisition, m=10: mean over 100 draws x 10 cohorts of (joint naming accuracy - strongest mean public learner accuracy) has '
                'crossed-bootstrap 95% lower bound > 0 (2000 replicates, seed 20261022)'),
            E2=('feature-directed acquisition, m=10, public only: accuracy of the selected public portfolio >= mean random-acquisition joint '
                'naming accuracy at m=10 (acquisition substitutes for private reports)'),
            success='E1 and E2 both hold'),
        predictions=['E1 holds','E2 holds',
            'partition ceilings (one-to-one, many-to-one, K=C) are lower than with DINOv2',
            'the random-acquisition joint gain at m=10 is smaller than with DINOv2 (+13.0)',
            'with feature-directed m=10 labels, the mean joint naming accuracy exceeds the directed public portfolio by at most 1 point'],
        descriptive=['all six label budgets for random acquisition','directed q=1,3,5','head-fixed native and joint control','partition ceilings'],
        reporting=dict(all_cases_retained=True,no_reruns_with_new_seeds=True,amendments='AMENDMENTS.json with time and reason'))

def main():
    path=ROOT/'ENCODER-DESIGN.json';d=json.loads(json.dumps(design()))
    if path.exists():
        assert json.loads(path.read_text())['design']==d,'frozen design differs';print('design unchanged');return
    rec=dict(design=d,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('*.py'))})
    path.write_text(json.dumps(rec,indent=2)+'\n');print('frozen',rec['frozen_utc'])

if __name__=='__main__':main()
