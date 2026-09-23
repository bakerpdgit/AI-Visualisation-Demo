// The five "taps" into the network that we show on screen, from shallow to deep.
// Each one is a layer whose output grid is half the size of the previous one.
// Text is written for a mixed open-morning audience (Year 6 pupils to parents).

export const STAGES = [
  {
    tap: 'stem',
    title: 'Colours & edges',
    short: 'Tiny 3×3 patches',
    blurb: 'Each detector looks at a tiny patch of pixels and finds simple things: edges, lines at different angles and blobs of colour.',
  },
  {
    tap: 'b3.expand',
    title: 'Simple patterns',
    short: 'Patches about 19 pixels wide',
    blurb: 'Edges get combined into corners, curves, stripes and spots.',
  },
  {
    tap: 'b5.expand',
    title: 'Textures',
    short: 'Patches about 67 pixels wide',
    blurb: 'Patterns get combined into textures, like fluffy fur, bumpy scales and whiskers.',
  },
  {
    tap: 'b11.expand',
    title: 'Parts',
    short: 'Most of the picture',
    blurb: 'Textures get combined into parts of an animal: eyes, ears, noses, teeth and snouts.',
  },
  {
    tap: 'head',
    title: 'Whole animals',
    short: 'The whole picture',
    blurb: 'Detectors here respond to big ideas like "cat face" or "scaly reptile". The last step adds up their votes.',
  },
];

export const TILES_PER_STAGE = 6;
export const TILES_IN_DETAIL = 40;
