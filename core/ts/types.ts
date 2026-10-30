export type Triple = { min: number; typ: number; max: number };
export type SugarLevel = 'full' | 'less' | 'half' | 'light' | 'none';
export interface DishEntry {
  id: string;
  kind: 'solid' | 'drink';
  density_g_ml?: Triple;
  kcal_100g: Triple;
  oil_g_per_100g?: { min: number; max: number };
  default_portion_g?: Triple;
}
export interface DrinkEntry {
  id: string;
  kind?: 'drink';
  base_kcal_per_700ml: Triple;
  toppings_kcal: Triple;
}
export interface ItemEstimate { id: string; lo: number; med: number; hi: number; flags: string[] }
export interface MealRange { lo: number; med: number; hi: number }
