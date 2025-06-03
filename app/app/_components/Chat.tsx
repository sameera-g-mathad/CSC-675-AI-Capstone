import { StyleSheet, Text, View } from "react-native";
import React, { memo, PropsWithChildren } from "react";

interface chatInterface {
    role: 'user' | 'bot'
}
const colorsToUse = {
    'user': {
        borderColor: '#7dd3fc',
        backgroundColor: '#e0f2fe'
    },
    'bot': {
        borderColor: '#f9a8d4',
        backgroundColor: '#ffe4e6'
    }
}
function Chat({ children, role }: PropsWithChildren & chatInterface) {
    const { borderColor, backgroundColor } = colorsToUse[role];
    return (
        <View style={styles.viewContainer}>
            <View style={{ ...styles.msgContainer, borderLeftColor: borderColor, backgroundColor }}>
                {children}
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    viewContainer: {
        margin: 10
    },
    msgContainer: {
        padding: 10,
        borderLeftWidth: 5,
        backgroundColor: '#e2e8f0',
    },
    timeContainer: {
        padding: 5
    }
});


export default memo(Chat)