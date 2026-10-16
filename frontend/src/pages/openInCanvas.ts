import { api } from '../api/client'
import { useStudio, type CanvasSeed } from '../store/studio'

/** Seed the canvas; returns false when the user keeps the current graph. */
async function seedCanvas(seed: CanvasSeed): Promise<boolean> {
  const studio = useStudio.getState()
  if (studio.nodes.length > 0 && !window.confirm('Replace the current canvas with a new Image → Model → Preview graph? Unsaved changes will be lost.')) return false
  studio.setWeights(await api.weights()) // downloads may have registered new weights since startup
  useStudio.getState().openInCanvas(seed)
  return true
}

/** Open one image of an imported dataset (the first previewable one by default) in the canvas. */
export async function openDatasetSample(datasetId: string, itemId?: string): Promise<boolean> {
  const id = itemId ?? (await api.datasetSamples(datasetId, 1))[0]?.item_id
  if (!id) throw new Error('This dataset has no previewable images')
  const image = await api.importDatasetSample(datasetId, id)
  return seedCanvas({ image })
}

/** Why an imported model cannot be opened directly, or undefined when it can. Models without a weight use the Choose weights dialog. */
export function modelUnavailableReason(artifact: any): string | undefined {
  if (!artifact) return 'Model details are still loading'
  if (artifact.weight_id) return undefined
  return `${artifact.status}: ${artifact.inspection?.explanation || 'no compatible weight was registered'}`
}

/** Open a registered weight of a known family in the canvas. */
export function openWeight(family: string, weightId: string, hint?: string): Promise<boolean> {
  return seedCanvas({ model: { family, weightId, hint } })
}

/** Open an imported model whose artifact was registered as a weight in the canvas. */
export async function openModelArtifact(artifact: any): Promise<boolean> {
  const reason = modelUnavailableReason(artifact)
  if (reason) throw new Error(reason)
  const weight = (await api.weights()).find((item) => item.weight_id === artifact.weight_id)
  return seedCanvas({ model: { family: weight?.family || artifact.candidate_adapter, weightId: artifact.weight_id, hint: `${artifact.artifact_file || ''} ${artifact.architecture || ''} ${artifact.display_name || ''}` } })
}
