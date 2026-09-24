import { PREDEFINED_PHRASES } from './data/phrases.js';
const NOT_FOUND_MESSAGE = "Phrase not available in the current sign library.";
function normalizeInput(input) {
  if (!input) return "";
  return input.toLowerCase().trim().replace(/[\u2018\u2019`]/g, "'").replace(/'/g, "").replace(/[?!.,;:"()[\]{}_+\-–—/*#@$%^&~]/g, " ").replace(/\s+/g, " ").trim();
}
function matchPhrase(input) {
  const trimmed = (input || "").trim();
  if (!trimmed) {
    return {
      matched: false,
      phrase: null,
      query: input,
      message: "Please enter or select a phrase to find its sign animation."
    };
  }
  const exactMatch = PREDEFINED_PHRASES.find(
    (p) => p.displayText.trim() === trimmed
  );
  if (exactMatch) {
    return {
      matched: true,
      phrase: exactMatch,
      query: input,
      message: `Matched phrase: "${exactMatch.displayText}"`
    };
  }
  const normalizedQuery = normalizeInput(trimmed);
  const normalizedMatch = PREDEFINED_PHRASES.find((p) => {
    const normStored = normalizeInput(p.displayText);
    return normStored === normalizedQuery;
  });
  if (normalizedMatch) {
    return {
      matched: true,
      phrase: normalizedMatch,
      query: input,
      message: `Matched phrase: "${normalizedMatch.displayText}"`
    };
  }
  return {
    matched: false,
    phrase: null,
    query: input,
    message: NOT_FOUND_MESSAGE
  };
}
function getAllPhrases() {
  return PREDEFINED_PHRASES;
}
function getPhraseById(id) {
  return PREDEFINED_PHRASES.find((p) => p.id === id);
}
function searchPhrases(filterQuery) {
  if (!filterQuery || !filterQuery.trim()) {
    return PREDEFINED_PHRASES;
  }
  const norm = normalizeInput(filterQuery);
  return PREDEFINED_PHRASES.filter((p) => {
    return normalizeInput(p.displayText).includes(norm) || p.category.toLowerCase().includes(norm) || p.slug.includes(norm);
  });
}
export {
  NOT_FOUND_MESSAGE,
  getAllPhrases,
  getPhraseById,
  matchPhrase,
  normalizeInput,
  searchPhrases
};
