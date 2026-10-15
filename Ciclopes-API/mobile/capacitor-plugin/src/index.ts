import { registerPlugin } from '@capacitor/core';
import type { CiclopesLaneAnalysisPlugin } from './definitions';

export const CiclopesLaneAnalysis =
  registerPlugin<CiclopesLaneAnalysisPlugin>('CiclopesLaneAnalysis');

export * from './definitions';
