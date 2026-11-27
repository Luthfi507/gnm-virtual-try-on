/**
 * Character appearance presets and GNM component palettes (Option A customizer).
 */

export interface ComponentPalette {
  deep: number;
  lit: number;
  fill: number;
  fillStrength: number;
  rim: number;
  rimPower: number;
  rimStrength: number;
  specular: number;
  emissive: number;
  emissiveStrength: number;
}

export interface CharacterStyle {
  presetId: string;
  skinDeep: string;
  skinLit: string;
  skinFill: string;
  skinRim: string;
  eyeGlow: string;
  wireColor: string;
  wireOpacity: number;
  wireVisible: boolean;
  bloomStrength: number;
  fillStrength: number;
  rimStrength: number;
  specular: number;
  background: string;
}

export interface CharacterIdentityPreset {
  id: string;
  name: string;
  identity: number[];
}

export const STYLE_PRESETS: Record<string, { label: string; style: CharacterStyle }> = {
  cyber_teal: {
    label: 'Cyber Teal',
    style: {
      presetId: 'cyber_teal',
      skinDeep: '#1a2350',
      skinLit: '#2f8871',
      skinFill: '#9b3ca1',
      skinRim: '#54c4a7',
      eyeGlow: '#2f8871',
      wireColor: '#7cdec5',
      wireOpacity: 0.35,
      wireVisible: true,
      bloomStrength: 0.38,
      fillStrength: 0.85,
      rimStrength: 0.95,
      specular: 0.35,
      background: '#070a18',
    },
  },
  studio_clay: {
    label: 'Studio Clay',
    style: {
      presetId: 'studio_clay',
      skinDeep: '#3a2c29',
      skinLit: '#d8a68e',
      skinFill: '#7b89a6',
      skinRim: '#f3d2be',
      eyeGlow: '#1a1a1a',
      wireColor: '#ffffff',
      wireOpacity: 0.08,
      wireVisible: false,
      bloomStrength: 0.12,
      fillStrength: 0.55,
      rimStrength: 0.45,
      specular: 0.22,
      background: '#11131a',
    },
  },
  metallic_gold: {
    label: 'Gilded Bronze',
    style: {
      presetId: 'metallic_gold',
      skinDeep: '#24180c',
      skinLit: '#d49a45',
      skinFill: '#3b6d8c',
      skinRim: '#ffe09c',
      eyeGlow: '#52d1ff',
      wireColor: '#ffd27a',
      wireOpacity: 0.22,
      wireVisible: true,
      bloomStrength: 0.45,
      fillStrength: 0.65,
      rimStrength: 1.15,
      specular: 1.55,
      background: '#0a090d',
    },
  },
  neon_synth: {
    label: 'Neon Synth',
    style: {
      presetId: 'neon_synth',
      skinDeep: '#120824',
      skinLit: '#34225e',
      skinFill: '#ff3b9a',
      skinRim: '#00f0ff',
      eyeGlow: '#ff2e93',
      wireColor: '#00f0ff',
      wireOpacity: 0.65,
      wireVisible: true,
      bloomStrength: 0.72,
      fillStrength: 1.1,
      rimStrength: 1.4,
      specular: 0.9,
      background: '#05030c',
    },
  },
};

export function hexToNumber(hex: string): number {
  return parseInt(hex.replace('#', ''), 16);
}

export function buildComponentPalettes(style: CharacterStyle): ComponentPalette[] {
  const skinDeep = hexToNumber(style.skinDeep);
  const skinLit = hexToNumber(style.skinLit);
  const skinFill = hexToNumber(style.skinFill);
  const skinRim = hexToNumber(style.skinRim);
  const eyeGlow = hexToNumber(style.eyeGlow);

  return [
    // 0: skin
    {
      deep: skinDeep,
      lit: skinLit,
      fill: skinFill,
      fillStrength: style.fillStrength,
      rim: skinRim,
      rimPower: 3.4,
      rimStrength: style.rimStrength,
      specular: style.specular,
      emissive: 0x000000,
      emissiveStrength: 0,
    },
    // 1: left_eye
    {
      deep: 0x0b0f24,
      lit: eyeGlow,
      fill: skinFill,
      fillStrength: 0.2,
      rim: skinRim,
      rimPower: 2.0,
      rimStrength: 1.4,
      specular: 1.6,
      emissive: eyeGlow,
      emissiveStrength: 0.32,
    },
    // 2: right_eye
    {
      deep: 0x0b0f24,
      lit: eyeGlow,
      fill: skinFill,
      fillStrength: 0.2,
      rim: skinRim,
      rimPower: 2.0,
      rimStrength: 1.4,
      specular: 1.6,
      emissive: eyeGlow,
      emissiveStrength: 0.32,
    },
    // 3: upper_teeth_and_gums
    {
      deep: skinDeep,
      lit: 0xd9e4df,
      fill: skinFill,
      fillStrength: 0.12,
      rim: 0xf0f6f3,
      rimPower: 3.6,
      rimStrength: 0.5,
      specular: 0.5,
      emissive: 0x000000,
      emissiveStrength: 0,
    },
    // 4: lower_teeth_and_gums
    {
      deep: skinDeep,
      lit: 0xd9e4df,
      fill: skinFill,
      fillStrength: 0.12,
      rim: 0xf0f6f3,
      rimPower: 3.6,
      rimStrength: 0.5,
      specular: 0.5,
      emissive: 0x000000,
      emissiveStrength: 0,
    },
    // 5: tongue
    {
      deep: 0x1a0a26,
      lit: 0x9c487c,
      fill: skinFill,
      fillStrength: 0.4,
      rim: 0xc06fc4,
      rimPower: 2.6,
      rimStrength: 0.75,
      specular: 0.65,
      emissive: 0x000000,
      emissiveStrength: 0,
    },
  ];
}
