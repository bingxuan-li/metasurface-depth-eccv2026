import Image from './components/image';

const opticalResults = [
  {
    file: 'point-source-precision',
    title: 'Point Sources and Photon Budget',
    alt: 'Depth precision bounds for our PSF, DeepDfD, double-helix and conventional lens designs at three photon budgets.',
    caption: 'A point of light is imaged at different distances with four PSF designs. The vertical axis is the Cramér–Rao bound (CRLB): a theoretical lower limit on the standard deviation of an unbiased depth estimate, not measured model error. Lower values on this logarithmic axis mean better attainable precision. From left to right, the photon budget falls from 100,000 to 10,000 to 1,000. The rotating designs avoid the conventional lens’s near-focus spike and vary smoothly with depth; DeepDfD’s relative performance depends on distance and photon budget.',
  },
  {
    file: 'edge-orientation',
    title: 'Edge Orientation',
    alt: 'Heatmaps of the depth precision bound versus distance and edge orientation for our PSF, double-helix and DeepDfD designs.',
    caption: 'A straight bright–dark edge models an object boundary. The horizontal axis is its distance and the vertical axis its orientation in radians; both are unknown in this calculation. Purple means a lower depth-uncertainty bound, yellow a higher one. Our polarization-separated rotating PSF varies smoothly across orientations, whereas the double-helix design has pronounced bands of poorer precision. Separating the lobes therefore reduces this orientation-dependent ambiguity. DeepDfD is nearly orientation-independent because its PSF is radially symmetric.',
  },
  {
    file: 'edge-precision',
    title: 'Extended Edges and Photon Budget',
    alt: 'Orientation-averaged edge depth precision bounds at three photon budgets for three PSF designs.',
    caption: 'These curves average the edge depth-uncertainty bound over orientation. Lower is better. From left to right, the photon budgets are one million, 100,000, and 10,000; each panel has a different vertical scale. Our design (red) has a lower mean bound than the double-helix PSF (green) throughout the plotted range, supporting the benefit of separating its two lobes into polarization channels. The comparison with DeepDfD (blue) depends on depth and photon budget; no design is best everywhere.',
  },
  {
    file: 'depth-correlation',
    title: 'Ambiguity Across Depths',
    alt: 'Pairwise PSF correlation matrices over one to five meters for our PSF, double-helix and DeepDfD designs.',
    caption: 'Each pixel compares the PSFs at two distances, read from the horizontal and vertical axes. Yellow and white mean high similarity; bright regions away from the diagonal indicate depths that can be confused. Both rotating designs concentrate similarity near the diagonal, unlike DeepDfD’s broadly similar patterns. The reported condition number describes how sensitive optical depth recovery is to measurement errors. Ours is about one-seventh of DeepDfD’s, supporting more stable recovery in this theoretical setting.',
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
            Unequal brightness in the two polarization channels can weaken
            the depth cue. We simulate this on Hypersim by varying the degree of
            linear polarization (DoLP) and its angle (AoLP). Bar height shows
            mean absolute depth error in centimetres; lower is better. Error
            stays comparatively stable at moderate polarization, then rises
            at high DoLP, especially near 0° and 90°, where one channel becomes
            much dimmer. This tests global channel imbalance, not every form
            of spatially varying reflection or polarization.
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
            <caption>
              Mean absolute depth error (MAE), in metres; lower is better.
              Fine-tuning the Small UniDepth V2 model on our polarization input
              gives lower error than the original Large model on all three
              datasets. This shows that the input adaptation also works with a
              second depth-model family, rather than depending only on Depth
              Anything V2. Results are reproduced from the supplementary
              material.
            </caption>
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
        <h2>PSF Analysis</h2>
        <p className="section-intro">
          A point-spread function (PSF) is the image formed by a point of light.
          Our design makes it rotate with distance and records opposite lobes
          in separate polarization channels. These theoretical comparisons use
          a 1–5 m range and 50 mm focal length, not the near-range prototype.
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
