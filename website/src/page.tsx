import Image from './components/image';
import { FileText, Code2, Database, Layers3, Play } from 'lucide-react';
import { PublicResultsGallery } from './results-explorer';
import { SupplementaryResults } from './supplementary-results';

const authors = [
  ['Bingxuan Li', '*,1'],
  ['Jiahao Wu', '*,2'],
  ['Yuan Xu', '*,2'],
  ['Zezheng Zhu', '2'],
  ['Yunxiang Zhang', '1'],
  ['Kenneth Chen', '1'],
  ['Yanqi Liang', '2'],
  ['Nanfang Yu', '†,2'],
  ['Qi Sun', '†,1'],
];

export default function Home() {
  return (
    <>
      <a href="#abstract" className="skip-link">
        Skip to abstract
      </a>
      <main id="top">
        <header className="paper-header">
          <h1>
            Physically Grounded Monocular Depth
            <br className="title-break" /> via Nanophotonic Wavefront Encoding
          </h1>
          <div className="authors">
            {authors.map(([name, note]) => (
              <span key={name}>
                {name}
                <sup>{note}</sup>
              </span>
            ))}
          </div>
          <p className="institutions">
            <span>
              <sup>1</sup> New York University
            </span>
            <span>
              <sup>2</sup> Columbia University
            </span>
          </p>
          <p className="contributions">
            * Equal contribution &nbsp; † Corresponding authors
          </p>
          <p className="venue">ECCV 2026</p>
          <nav className="paper-links" aria-label="Paper resources">
            <a
              href="https://arxiv.org/abs/2503.15770"
              target="_blank"
              rel="noreferrer"
            >
              <FileText size={17} />
              Paper
            </a>
            <a
              href="https://github.com/bingxuan-li/metasurface-depth-eccv2026"
              target="_blank"
              rel="noreferrer"
            >
              <Code2 size={17} />
              Code
            </a>
            <a href="#video">
              <Play size={17} />
              Video
            </a>
            <a
              href="https://huggingface.co/Bingxuan111/metasurface-depth-eccv2026"
              target="_blank"
              rel="noreferrer"
            >
              <Layers3 size={17} />
              Models
            </a>
            <a href="https://huggingface.co/datasets/Bingxuan111/metasurface-real-eccv2026" target="_blank" rel="noreferrer">
              <Database size={17} />
              Dataset
            </a>
          </nav>
        </header>

        <figure className="teaser wide">
          <a
            href="./figures/teaser.png"
            target="_blank"
            rel="noreferrer"
            aria-label="Open the system overview at full resolution"
          >
            <Image
              unoptimized
              priority
              src="./figures/teaser.png"
              width={2200}
              height={873}
              alt="System overview: a birefringent metalens produces two polarization views; a fine-tuned depth foundation model recovers metric depth. The fabricated lens and nanopillars are shown below."
            />
          </a>
          <figcaption>
            A birefringent metalens encodes depth in two polarization channels.
          </figcaption>
        </figure>

        <section className="text-section" id="abstract">
          <h2>Overview</h2>
          <p>
            We combine a depth-encoding metalens with a pretrained depth model.
            The metalens produces two images whose relative PSF shifts depend on
            scene depth. We fine-tune the model on simulated image pairs and
            five real captures to estimate metric depth from a single shot.
          </p>
        </section>

        <section className="figure-section" id="video">
          <h2>Video</h2>
          <div className="wide">
            <iframe
              src="https://www.youtube-nocookie.com/embed/Qkb4nXjKlwU"
              title="Physically Grounded Monocular Depth — ECCV 2026 video"
              style={{ width: '100%', aspectRatio: '16 / 9', border: 0 }}
              loading="lazy"
              referrerPolicy="strict-origin-when-cross-origin"
              allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen"
              allowFullScreen
            />
          </div>
        </section>

        <section className="figure-section" id="simulation-results">
          <h2>Simulation Results</h2>
          <p className="section-intro">
            NYU Depth V2 · MPI Sintel · Hypersim · MIT-CGH-4K
          </p>

          <figure className="wide comparison-figure">
            <a
              href="./figures/simulation-more.png"
              target="_blank"
              rel="noreferrer"
              aria-label="Enlarge ten simulated scenes across four datasets"
            >
              <Image
                unoptimized
                src="./figures/simulation-more.png"
                width={2595}
                height={3000}
                loading="lazy"
                alt="Ten simulated comparisons: three NYU Depth V2, three MPI Sintel, two Hypersim test, and two MIT-CGH-4K scenes. Columns show inputs, GT, ours, UniDepth V2, DepthPro, Depth Anything V2 and Marigold."
              />
            </a>
            <figcaption>
              Comparisons across four datasets. Baselines are aligned to ground
              truth in scale and shift; our predictions are unaligned.
            </figcaption>
          </figure>
          <figure className="wide comparison-figure">
            <h3 className="simulation-subheading">
              FlyingThings3D · Depth-prior ablation
            </h3>
            <a
              href="./figures/simulation-ablation.png"
              target="_blank"
              rel="noreferrer"
              aria-label="Enlarge the FlyingThings3D depth-prior ablation"
            >
              <Image
                unoptimized
                src="./figures/simulation-ablation.png"
                width={2713}
                height={1018}
                loading="lazy"
                alt="Three FlyingThings3D scenes comparing simulated inputs and ground truth with the full model, no pretrained initialization, and a U-Net backbone."
              />
            </a>
            <figcaption>
              Effect of pretrained depth priors: the full model, training
              without pretraining, and a U-Net backbone.
            </figcaption>
          </figure>
        </section>

        <PublicResultsGallery />

        <section className="figure-section" id="results">
          <h2>Real-World Comparisons</h2>
          <p className="section-intro">
            Single- and multi-object scenes captured with the prototype.
          </p>

          <figure className="wide comparison-figure">
            <a
              href="./figures/physical-comparison.png"
              target="_blank"
              rel="noreferrer"
            >
              <Image
                unoptimized
                src="./figures/physical-comparison.png"
                width={1982}
                height={2400}
                loading="lazy"
                alt="Eleven physical scenes comparing our paper results with Depth Anything V2, UniDepth V2, DepthPro and Marigold, alongside inputs and depth labels."
              />
            </a>
            <figcaption>
              Comparisons on eleven real scenes. Depth Anything V2* is
              fine-tuned; other baselines are aligned to ground truth. Insets
              show errors.
            </figcaption>
          </figure>
        </section>

        <section className="figure-section" id="depth-consistency">
          <h2>Depth Consistency</h2>
          <figure className="wide">
            <a
              href="./figures/depth-consistency.png"
              target="_blank"
              rel="noreferrer"
              aria-label="Enlarge depth predictions for moving objects"
            >
              <Image
                unoptimized
                src="./figures/depth-consistency.png"
                width={2191}
                height={1128}
                loading="lazy"
                alt="Six frames of input images and predicted depth: a simulated moving figure above and a real cat moving toward the camera below."
              />
            </a>
            <figcaption>
              Depth predictions as objects move toward the camera. Top:
              simulated sequence. Bottom: real captures. Each pair of rows shows
              the input and predicted depth.
            </figcaption>
          </figure>
        </section>

        <section className="text-section" id="method">
          <h2>Method</h2>
          <p>
            Our optical simulator maps RGB-D scenes to polarization image pairs.
            Both simulated and captured pairs are stacked as{' '}
            <span className="math">
              [I<sub>x</sub>, I<sub>y</sub>, (I<sub>x</sub> + I<sub>y</sub>) /
              2]
            </span>{' '}
            and passed to the depth model.
          </p>
        </section>
        <figure className="wide method-figure">
          <a
            href="./figures/pipeline.png"
            target="_blank"
            rel="noreferrer"
            aria-label="Open the learning and simulation pipeline at full resolution"
          >
            <Image
              unoptimized
              src="./figures/pipeline.png"
              width={2200}
              height={1028}
              loading="lazy"
              alt="Full pipeline: RGB-D simulation, augmentation, three-channel input adaptation and depth prediction; the lower portion details the optical forward model."
            />
          </a>
          <figcaption>Training pipeline and optical forward model.</figcaption>
        </figure>

        <section className="figure-section" id="simulator-results">
          <h2>Optical Simulation</h2>
          <figure className="wide">
            <a href="./figures/simulator-ablation.png" target="_blank" rel="noreferrer" aria-label="Enlarge the optical simulator comparison">
              <Image src="./figures/simulator-ablation.png" width={2400} height={1107} loading="lazy"
                alt="Linear convolution and our optical forward model, with sphere-boundary and indoor-scene ablations of disocclusion handling." />
            </a>
            <figcaption>
              Disocclusion handling reduces boundary artifacts in the simulated polarization images.
              These are optical-rendering comparisons, not depth predictions.
            </figcaption>
          </figure>
        </section>

        <SupplementaryResults />

        <section className="text-section" id="resources">
          <h2>Code, Models, and Data</h2>
          <p>
            The release includes the optical simulator, training and inference
            code, and Small, Base, and Large checkpoints. The real-capture
            dataset contains five training scenes and 42 evaluation scenes.
            Inputs and depth labels are available under{' '}
            <a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a>{' '}
            for noncommercial use, including academic research.
          </p>
          <details className="technical-notes">
            <summary>Evaluation protocol and limitations</summary>
            <p>
              The 42 scenes were also used for validation and model selection,
              not as an untouched holdout. Their depth labels are manually
              segmented regions assigned measured object distances, not dense 3D
              scans. The explorer uses the selected mixed-training Large
              checkpoint; paper figures retain their original results and are
              not a fresh benchmark.
            </p>
            <p>
              The prototype targets well-lit, near-range scenes. Limited light
              throughput, field of view, and high-frequency polarization
              imbalance remain limitations. Code is MIT-licensed with retained
              third-party notices. Small weights follow Apache-2.0; Base and
              Large follow CC-BY-NC-4.0.
            </p>
          </details>
        </section>

        <section className="text-section citation" id="bibtex">
          <h2>BibTeX</h2>
          <pre>
            <code>{`@article{li2026physically,
  title={Physically Grounded Monocular Depth via
         Nanophotonic Wavefront Encoding},
  author={Li, Bingxuan and Wu, Jiahao and Xu, Yuan and
          Zhu, Zezheng and Zhang, Yunxiang and Chen, Kenneth and
          Liang, Yanqi and Yu, Nanfang and Sun, Qi},
  journal={arXiv preprint arXiv:2503.15770},
  year={2026}
}`}</code>
          </pre>
        </section>
      </main>
      <footer>
        <p>ECCV 2026</p>
        <a href="https://github.com/bingxuan-li/metasurface-depth-eccv2026/tree/main/website">Website source</a>
        <a href="#top">Back to top</a>
      </footer>
    </>
  );
}
