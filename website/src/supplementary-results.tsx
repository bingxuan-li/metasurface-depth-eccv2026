import Image from './components/image';

const opticalResults = [
  {
    file: 'point-source-precision',
    title: 'Point Sources and Photon Budget',
    alt: 'Depth precision bounds for our PSF, DeepDfD, double-helix and conventional lens designs at three photon budgets.',
    caption: 'Point-source depth precision at three photon budgets. Lower bounds indicate higher attainable precision.',
  },
  {
    file: 'edge-orientation',
    title: 'Edge Orientation',
    alt: 'Heatmaps of the depth precision bound versus distance and edge orientation for our PSF, double-helix and DeepDfD designs.',
    caption: 'Depth sensitivity depends on both distance and edge direction. The heatmaps show the precision bound when both are unknown.',
  },
  {
    file: 'edge-precision',
    title: 'Extended Edges and Photon Budget',
    alt: 'Orientation-averaged edge depth precision bounds at three photon budgets for three PSF designs.',
    caption: 'Edge depth bounds averaged over orientation, under three photon budgets.',
  },
  {
    file: 'depth-correlation',
    title: 'Ambiguity Across Depths',
    alt: 'Pairwise PSF correlation matrices over one to five meters for our PSF, double-helix and DeepDfD designs.',
    caption: 'Cross-depth PSF correlation. Strong off-diagonal similarity makes different depths harder to distinguish.',
  },
];

export function SupplementaryResults() {
  return (
    <div id="supplementary-results">
      <section className="figure-section" id="polarization-robustness">
        <h2>Polarization Robustness</h2>
        <figure className="wide polarization-figure">
          <a href="./figures/polarization-robustness.png" target="_blank" rel="noreferrer" aria-label="Enlarge the polarization robustness experiment">
            <Image src="./figures/polarization-robustness.png" width={2400} height={1847} loading="lazy"
              alt="Hypersim depth error in centimeters as the degree and angle of linear polarization change." />
          </a>
          <figcaption>
            Hypersim under partially polarized illumination. Error rises at high
            degrees of polarization, particularly when one channel is strongly attenuated.
          </figcaption>
        </figure>
      </section>

      <section className="text-section" id="model-generalization">
        <h2>Transfer to UniDepth V2</h2>
        <p className="section-intro">
          The same three-channel input adaptation with a different depth model.
        </p>
        <div className="supplementary-table-wrap">
          <table className="supplementary-table">
            <caption>MAE (m) from the supplementary material. Lower is better.</caption>
            <thead>
              <tr>
                <th scope="col">Dataset</th>
                <th scope="col">Original Large</th>
                <th scope="col">Fine-tuned Small</th>
              </tr>
            </thead>
            <tbody>
              <tr><th scope="row">NYU Depth V2</th><td>0.0338</td><td><strong>0.0217</strong></td></tr>
              <tr><th scope="row">MIT-CGH-4K</th><td>0.1273</td><td><strong>0.0708</strong></td></tr>
              <tr><th scope="row">Real captures</th><td>0.1068</td><td><strong>0.0348</strong></td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="figure-section" id="depth-encoding">
        <h2>What the PSF Encodes</h2>
        <p className="section-intro">
          Theoretical comparisons over 1–5 m with a 50 mm focal length.
          These use a different optical configuration from the near-range prototype.
        </p>
        {opticalResults.map(({ file, title, alt, caption }) => (
          <figure className="wide comparison-figure" key={file}>
            <h3 className="simulation-subheading">{title}</h3>
            <a href={`./figures/${file}.png`} target="_blank" rel="noreferrer" aria-label={`Enlarge: ${title}`}>
              <Image src={`./figures/${file}.png`} width={2400} height={641} loading="lazy" alt={alt} />
            </a>
            <figcaption>{caption}</figcaption>
          </figure>
        ))}
      </section>
    </div>
  );
}
