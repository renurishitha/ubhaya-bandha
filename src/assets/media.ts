import heroBispecificOverview from './images/hero_bispecific_overview_1794582591737.jpg';
import mechBridgeCells from './images/mech_bridge_cells_1794582591334.jpg';
import mechImmunotherapy from './images/mech_immunotherapy_1794582591444.jpg';
import mechTaaDual from './images/mech_taa_dual_1794582591965.jpg';
import scRnaAtlas from './images/sc_rna_atlas_1794582593292.jpg';

export const BSPAI_MEDIA = {
  heroBispecificOverview,
  mechBridgeCells,
  mechTaaDual,
  mechImmunotherapy,
  scRnaAtlas,
} as const;

export function getMechanismImage(mechanismType?: string): {
  src: string;
  alt: string;
  caption: string;
  accentColor: string;
} {
  switch (mechanismType) {
    case 'bridge-two-cells':
      return {
        src: mechBridgeCells,
        alt: 'Bispecific antibody bridging a tumor-associated antigen and an immune effector T-cell receptor across two cells',
        caption: 'Trans-cellular bridging: links distinct cell compartments (e.g., epithelial TAA to T-cell CD3/checkpoint)',
        accentColor: '#2563EB',
      };
    case 'TAA+TAA':
      return {
        src: mechTaaDual,
        alt: 'Dual tumor-associated antigen bispecific antibody engaging two complementary membrane receptors on a cancer cell',
        caption: 'Cis dual-TAA blockade: co-targets two malignant cell-surface pathways (e.g., EGFR + MET) to curb escape',
        accentColor: '#7C3AED',
      };
    case 'tumor-immunotherapy':
      return {
        src: mechImmunotherapy,
        alt: 'Dual immune checkpoint blockade bispecific antibody targeting PD-L1 and CTLA-4 across the tumor-immune synapse',
        caption: 'Dual immunomodulation: relieves co-inhibitory signaling across exhausted T cells, Tregs, and tumor cells',
        accentColor: '#0D9488',
      };
    default:
      return {
        src: heroBispecificOverview,
        alt: 'Y-shaped bispecific antibody engaging two distinct cell-surface targets in the tumor microenvironment',
        caption: 'Bispecific target-pair evaluation combining scRNA-seq activity, tumor-normal safety, and pathway co-occurrence',
        accentColor: '#2563EB',
      };
  }
}
