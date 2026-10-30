import nutrition from '../../data/nutrition.json';
import labels from './generated/labels.json';
import { buildMealData } from './meal.ts';
import type { RawNutrition } from './meal.ts';

export const data = buildMealData(nutrition as unknown as RawNutrition);
/** Dish ids in model class order (ml/labels.yaml), restricted to those with nutrition data. */
export const dishIds: string[] = labels.filter(l => l.type === 'dish' && data.dishes[l.id]).map(l => l.id);
