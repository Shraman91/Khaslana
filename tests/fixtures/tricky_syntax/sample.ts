import { Request, Response } from 'express';
import type { Config, Logger as CustomLogger } from './types';

export interface Repository<T, ID extends string | number> {
    findById(id: ID): Promise<T | null>;
    saveAll(items: Array<T>): Promise<void>;
}

export class OrderService<T extends { id: string }, Metadata = Record<string, unknown>> {
    private logger: CustomLogger;

    constructor(logger: CustomLogger) {
        this.logger = logger;
    }

    public async processOrder<R = boolean>(
        orderId: string,
        payload: Map<string, T>,
        retries: number = 3,
    ): Promise<R> {
        this.logger.info("Processing order", orderId);
        const validated = validateOrder(orderId, payload);
        return executePayment(validated);
    }
}

export const calculateDiscount = async <T, U extends Record<string, number>>(
    basePrice: number,
    metadata: U,
    factor: number = 0.1,
): Promise<number> => {
    logTrace("calculating discount");
    return applyFactor(basePrice, factor);
};
