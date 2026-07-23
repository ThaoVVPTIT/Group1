`timescale 1ns / 1ps
// =============================================================
// FC3 : Lop ket noi day du 84 -> 47 (lop logit dau ra)
// KHONG dung luong tu hoa (mult/shift) va KHONG dung ReLU
// =============================================================
module fc3 (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    output wire        ready,
    output reg         done,

    input  wire        in_wr_en,
    input  wire [6:0]  in_wr_addr,
    input  wire signed [7:0] in_wr_data,

    // Dau ra la logit 32-bit, chua qua ReLU/quant
    input  wire [5:0]  out_rd_addr,
    output wire signed [31:0] out_rd_data
);

    reg signed [7:0]  f3_kernel [0:3947]; // 47 * 84
    reg signed [31:0] f3_bias   [0:46];

    initial begin
        $readmemh("weights_hex/fc3_kernel.hex", f3_kernel);
        $readmemh("weights_hex/fc3_bias.hex",   f3_bias);
    end

    reg signed [7:0]  fc2_out [0:83];
    reg signed [31:0] fc3_out [0:46];

    always @(posedge clk) begin
        if (in_wr_en) fc2_out[in_wr_addr] <= in_wr_data;
    end
    assign out_rd_data = fc3_out[out_rd_addr];

    localparam IDLE = 0, INIT = 1, MAC = 2, STORE = 3;
    reg [2:0] state;

    reg [15:0] c, kc;
    reg signed [31:0] acc;

    assign ready = (state == IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 0;
        end else begin
            done <= 0;
            case (state)
                IDLE: begin
                    if (start) begin
                        c <= 0;
                        state <= INIT;
                    end
                end

                INIT: begin
                    acc <= f3_bias[c];
                    kc <= 0;
                    state <= MAC;
                end

                MAC: begin
                    acc <= acc + fc2_out[kc] * f3_kernel[c*84 + kc];
                    if (kc == 83) state <= STORE;
                    else kc <= kc + 1;
                end

                STORE: begin
                    fc3_out[c] <= acc; // Khong luong tu hoa & khong ReLU
                    if (c == 46) begin
                        done  <= 1;
                        state <= IDLE;
                    end else begin
                        c <= c + 1;
                        state <= INIT;
                    end
                end
            endcase
        end
    end
endmodule